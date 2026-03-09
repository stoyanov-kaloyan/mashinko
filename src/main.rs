use arrayfire::{Array, af_print, dim4, randu};

fn main() {
    let x: Array<f32> = randu(dim4!(2, 2));
    let y: Array<f32> = randu(dim4!(2, 2));
    let xy = &x + &y;
    af_print!("f32arr", x);
    af_print!("f32arr", -y);
    af_print!("f32arr", xy);
}
