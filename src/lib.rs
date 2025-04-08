pub mod activator;
pub mod data_loader;
pub mod error;
pub mod nn;
pub mod prelude;

pub type Float = f64;

pub fn recognize_digit(nn: &nn::NN, grayscale_img: &[u8]) -> (usize, Float) {
    let input = grayscale_img
        .iter()
        .map(|&x| (x as Float) / 256.0)
        .collect::<Vec<Float>>();

    let output = nn.feedforward(&input);
    let mut max_idx = 0;

    for i in 1..9 {
        if output[i] > output[max_idx] {
            max_idx = i;
        }
    }

    (max_idx, output[max_idx])
}
