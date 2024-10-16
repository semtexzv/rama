use bytemuck::{cast_slice, from_bytes, Pod, Zeroable};
use cblas::{Layout, Transpose};

const MAGIC: [u8; 4] = *b"24ka";

#[derive(Debug, Clone, Copy, Zeroable, Pod)]
#[repr(C, packed)]
struct Header {
    pub ndim: u32,
    pub hdim: u32,
    pub nlay: u32,

    pub n_heads: u32,
    pub kv_heads: u32,
    pub vocab: u32,
    pub seq_len: u32,

    pub shared_classifier: u8,
}

struct Weights<'w, T: Pod = f32> {
    h: &'w Header,

    embed: &'w [T],

    rms_att: Vec<&'w [T]>,
    rms_ffn: Vec<&'w [T]>,

    wq: Vec<&'w [T]>,
    wk: Vec<&'w [T]>,
    wv: Vec<&'w [T]>,
    wo: Vec<&'w [T]>,

    w1: Vec<&'w [T]>,
    w2: Vec<&'w [T]>,
    w3: Vec<&'w [T]>,

    rms_fin: &'w [f32],

    wcls: &'w [f32],
}

fn split_at_many(cnt: u32, len: u32, mut data: &[f32]) -> (Vec<&[f32]>, &[f32]) {
    let cnt = cnt as usize;
    let len = len as usize;
    let mut out = vec![];
    for i in 0..cnt {
        let (a, b) = data.split_at(len);
        out.push(a);
        data = b;
    }
    (out, data)
}
#[inline(always)]
fn chunk_at(pos: u32, len: u32, mut data: &[f32]) -> &[f32] {
    &data[(pos * len) as usize..((pos + 1) * len) as usize]
}

#[inline(always)]
fn chunk_at_mut(pos: u32, len: u32, mut data: &mut [f32]) -> &mut [f32] {
    &mut data[(pos * len) as usize..((pos + 1) * len) as usize]
}

impl<'w> Weights<'w> {
    #[no_mangle]
    #[inline(never)]
    pub fn read(data: &'w [u8]) -> (Weights<'w>, &[u8]) {
        let (magic, data) = data.split_at(4);
        assert_eq!(MAGIC, magic);
        let (version, data) = data.split_at(4);
        assert_eq!(version, [2, 0, 0, 0]);
        let (h, data) = data.split_at(size_of::<Header>());

        let h: &Header = from_bytes(h);
        let head_size = h.ndim / h.n_heads;

        let data = &data[256 - (size_of::<Header>() + 8)..];
        let data: &[f32] = cast_slice(data);

        let (embed, data) = data.split_at((h.vocab * h.ndim) as usize);

        let (rms_att, data) = split_at_many(h.nlay, h.ndim, data);
        let (rms_ffn, data) = split_at_many(h.nlay, h.ndim, data);

        let (wq, data) = split_at_many(h.nlay, h.ndim * h.n_heads * head_size, data);
        let (wk, data) = split_at_many(h.nlay, h.ndim * h.kv_heads * head_size, data);
        let (wv, data) = split_at_many(h.nlay, h.ndim * h.kv_heads * head_size, data);
        let (wo, data) = split_at_many(h.nlay, h.ndim * h.n_heads * head_size, data);

        let (w1, data) = split_at_many(h.nlay, h.hdim * h.ndim, data);
        let (w2, data) = split_at_many(h.nlay, h.hdim * h.ndim, data);
        let (w3, data) = split_at_many(h.nlay, h.hdim * h.ndim, data);

        let (rms_fin, data) = data.split_at(h.ndim as usize);

        let (wcls, data) = if h.shared_classifier != 0 {
            data.split_at((h.vocab * h.ndim) as usize)
        } else {
            (embed, data)
        };

        (Weights {
            h,
            embed,
            rms_att,
            rms_ffn,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
            rms_fin,
            wcls,
        }, cast_slice(data))
    }
}

struct State {
    // activation at current time stamp (dim,)
    xa: Vec<f32>,
    // same, but inside a residual branch (dim,)
    xb: Vec<f32>,
    // an additional buffer just for convenience (dim,)
    xc: Vec<f32>,

    // buffer for hidden dimension in the ffn (hidden_dim,)
    hb: Vec<f32>,
    // buffer for hidden dimension in the ffn (hidden_dim,)
    hc: Vec<f32>,

    // query (dim,)
    q: Vec<f32>,

    // buffer for scores/attention values (n_heads, seq_len)
    att: Vec<f32>,
    // output logits
    log: Vec<f32>,
    // (layer, seq_len, dim)
    kc: Vec<f32>,
    // (layer, seq_len, dim)
    vc: Vec<f32>,
}

impl State {
    fn new(h: &Header) -> Self {
        Self {
            xa: vec![0.0; h.ndim as _],
            xb: vec![0.0; h.ndim as _],
            xc: vec![0.0; h.ndim as _],
            hb: vec![0.0; h.hdim as _],
            hc: vec![0.0; h.hdim as _],
            q: vec![0.0; h.ndim as _],

            att: vec![0.0; (h.n_heads * h.seq_len) as _],
            log: vec![0.0; h.vocab as _],
            kc: vec![0.0; (h.nlay * h.seq_len * h.ndim) as usize],
            vc: vec![0.0; (h.nlay * h.seq_len * h.ndim) as usize],
        }
    }
}
fn rmsnorm(o: &mut [f32], x: &[f32], w: &[f32]) {
    assert_eq!(o.len(), w.len());
    assert_eq!(x.len(), w.len());

    let mut ss: f32 = 0.0;
    for i in 0..x.len() {
        ss += x[i] * x[i];
    }
    ss /= x.len() as f32;
    ss += 1e-5;
    ss = 1.0 / ss.sqrt();
    unsafe {
        for i in 0..x.len() {
            o[i] = x[i] * w[i] * ss;
        }
    }
}

fn softmax(x: &mut [f32]) {
    let mut max_val = x[0];
    for i in 1..x.len() {
        if x[i] > max_val {
            max_val = x[i];
        }
    }
    let mut sum: f32 = 0.0;
    for i in 0..x.len() {
        x[i] = (x[i] - max_val).exp();
        sum += x[i];
    }
    for i in 0..x.len() {
        x[i] /= sum;
    }
}
fn argmax(x: &[f32]) -> usize {
    let mut max = f32::MIN;
    let mut pos = 0;
    for i in 1 .. x.len() {
        if x[i] > max {
            pos = i;
            max = x[i];
        }
    }
    pos
}

pub fn matmul(o: &mut [f32], x: &[f32], w: &[f32], n: u32, d: u32) {
    assert_eq!(o.len(), d as usize);
    assert_eq!(x.len(), n as usize);
    assert_eq!(w.len(), (n * d) as usize);

    // o.iter_mut().for_each(|w| *w = 0.0);
    // eprintln!("{:?} <- {:?} @ {:?}", o.as_ptr(), x.as_ptr(), w.as_ptr());
    // unsafe {
    //     cblas::sgemm(
    //         Layout::RowMajor,
    //         Transpose::None,
    //         Transpose::Ordinary,
    //         d as _,
    //         n as _,
    //         1,
    //         1.0,
    //         x,
    //         n as _,
    //         w,
    //         n as _,
    //         0.0,
    //         o,
    //         n as _,
    //     )
    // }
    // //     cblas::sgemm(
    // //         d,
    // //         n,
    // //         1,
    // //         1.0,
    // //         w.as_ptr(),
    // //         n as isize,
    // //         1,
    // //         x.as_ptr(),
    // //         1,
    // //         n as isize,
    // //         0.0,
    // //         o,
    // //         1,
    // //         1);
    // // }
    //
    for i in 0..d {
        let mut val: f32 = 0.0;
        for j in 0..n {
            val += w[(i * n + j) as usize] * x[j as usize];
        }
        unsafe {
            o[i as usize] = val;
        }
    }
}

fn forward(w: &Weights, s: &mut State, tok: u32, pos: u32) {
    let c = w.h;
    let kv_dim = (c.ndim * c.kv_heads) / c.n_heads;
    let kv_mul = c.n_heads / c.kv_heads;
    // panic!("{:?}", &w.w1[0][0..10]);
    let head_size = c.ndim / c.n_heads;

    s.xa.copy_from_slice(chunk_at(tok, c.ndim, w.embed));
    for l in 0..c.nlay as usize {
        // Layer key cache
        let lkc = chunk_at_mut(l as _, c.seq_len * kv_dim, &mut s.kc);
        // Layer value cache
        let lvc = chunk_at_mut(l as _, c.seq_len * kv_dim, &mut s.vc);

        rmsnorm(&mut s.xb, &s.xa, w.rms_att[l]);

        matmul(&mut s.q, &s.xb, &w.wq[l], c.ndim, c.ndim);
        matmul(chunk_at_mut(pos, kv_dim, lkc), &s.xb, &w.wk[l], c.ndim, kv_dim);
        matmul(chunk_at_mut(pos, kv_dim, lvc), &s.xb, &w.wv[l], c.ndim, kv_dim);

        for i in (0..c.ndim).step_by(2) {
            let head_dim = i % head_size;
            let freq = 1.9 / f32::powf(10000.0, (head_dim / head_size) as f32);
            let val = pos as f32 * freq;
            let fcr = f32::cos(val);
            let fci = f32::sin(val);
            let rotn = if i < kv_dim { 2 } else { 1 };
            for v in 0..rotn {
                let vec = if v == 0 { &mut s.q[..] } else { chunk_at_mut(pos, kv_dim, lkc) };
                let v0 = vec[i as usize];
                let v1 = vec[i as usize + 1];
                vec[i as usize] = v0 * fcr - v1 * fci;
                vec[i as usize + 1] = v0 * fci + v1 * fcr;
            }
        }


        for h in 0..c.n_heads {
            let q = chunk_at(h, head_size, &s.q);
            let att = chunk_at_mut(h, c.seq_len, &mut s.att);

            for t in 0..=pos {
                let k = chunk_at_mut(h / kv_mul, head_size, chunk_at_mut(t, kv_dim, lkc));
                let mut score = 0.0;
                for i in 0..head_size as usize {
                    score += q[i] * k[i];
                }
                score /= f32::sqrt(head_size as _);
                att[t as usize] = score;
            }

            softmax(&mut att[..pos as usize + 1]);

            let xb = chunk_at_mut(h, head_size, &mut s.xb);
            xb.iter_mut().for_each(|v| *v = 0.0);

            for t in 0..=pos {
                let v = chunk_at_mut(h / kv_mul, head_size, chunk_at_mut(t, kv_dim, lvc));
                let a = att[t as usize];
                for i in 0..head_size as usize {
                    xb[i] += a * v[i];
                }
            }
        }
        matmul(&mut s.xc, &s.xb, w.wo[l], c.ndim, c.ndim);

        for i in 0..c.ndim as usize {
            s.xa[i] += s.xc[i];
        }

        rmsnorm(&mut s.xb, &s.xa, w.rms_ffn[l]);

        matmul(&mut s.hb, &s.xb, w.w1[l], c.ndim, c.hdim);
        matmul(&mut s.hc, &s.xb, w.w3[l], c.ndim, c.hdim);
        for i in 0..c.hdim as usize {
            let mut val = s.hb[i];
            val *= (1.0 / (1.0 + f32::exp(-val)));
            val *= s.hc[i];
            s.hb[i] = val;
        }
        matmul(&mut s.xb, &s.hb, w.w2[l], c.hdim, c.ndim);
        for i in 0..c.ndim as usize {
            s.xa[i] += s.xb[i];
        }
    }
    rmsnorm(&mut s.xb, &s.xa, w.rms_fin);
    matmul(&mut s.log, &s.xb, w.embed, c.ndim, c.vocab);
}

fn main() -> anyhow::Result<()> {
    let file = std::fs::OpenOptions::new().read(true).open("data.bin")?;
    let file = unsafe { memmap2::Mmap::map(&file) }?;
    let (weights, rest) = Weights::read(&file);
    let mut state = State::new(weights.h);

    let mut tok = vec![1u32];
    loop {
        forward(&weights, &mut state, tok[tok.len() - 1], tok.len() as u32 - 1);
        let next = argmax(&state.log) as u32;
        println!("Generated {next} {tok:?}");
        if next == 1 {
            panic!("{:?}", tok)
        }

        tok.push(next);
    }
}
