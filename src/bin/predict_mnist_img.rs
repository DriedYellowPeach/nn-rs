use mnist::*;
use ndarray::prelude::*;
use nn_rs::prelude::*;
use rand::Rng;

const IMG_WIDTH: usize = 28;

fn main() {
    let val_size = 10_000;
    let Mnist {
        val_img, val_lbl, ..
    } = MnistBuilder::new()
        .base_path("data/") // Comment out this and `use_fashion_data()` to run on the original MNIST
        .label_format_digit()
        .training_set_length(val_size as u32)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let trn_data = Array2::from_shape_vec((val_size, IMG_WIDTH * IMG_WIDTH), val_img)
        .expect("Error converting images to Array3 struct");
    let nn = NNBuilder::new_from_model_file("data/model_9502.bin")
        .unwrap()
        .build()
        .unwrap();
    let mut rng = rand::rng();

    for _i in 0..10 {
        let img_idx = rng.random_range(0..val_size) as usize;
        let input_1d = trn_data.slice(s![img_idx, ..]).to_owned();
        let (digit, conf) = nn_rs::recognize_digit(&nn, &input_1d.to_vec());

        ascii_print(input_1d.view());
        println!("Recognized digit: {}, confidence: {}", digit, conf);
        println!("The digit is labeled as: {}", val_lbl[img_idx]);
        println!();
    }
}

fn ascii_print(img: ArrayView1<u8>) {
    for (idx, &v) in img.iter().enumerate() {
        if v < 128 {
            print!(".");
        } else {
            print!("&");
        }

        if (idx + 1) % IMG_WIDTH == 0 {
            println!();
        }
    }
}
