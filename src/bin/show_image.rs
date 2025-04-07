use image::*;
use mnist::*;
use ndarray::prelude::*;
use nn_rs::nn::NNBuilder;
use rand::Rng;
use show_image::{ImageInfo, ImageView, event};

#[show_image::main]
fn main() {
    env_logger::init();
    let (trn_size, _rows, _cols) = (50_000, 28, 28);

    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .base_path("data/") // Comment out this and `use_fashion_data()` to run on the original MNIST
        .label_format_digit()
        .training_set_length(trn_size)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    log::info!("trn_img with length: {}", trn_img.len());
    log::info!("trn_lbl with length: {}", trn_lbl.len());

    let item_num = rand::rng().random_range(0..trn_size) as usize;
    return_item_description_from_number(trn_lbl[item_num]);

    let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .mapv(|x| x as f32 / 256.);

    let input_2d = train_data.slice(s![item_num, .., ..]).to_owned();
    let image = bw_ndarray2_to_rgb_image(input_2d.clone());
    let input_1d = input_2d.to_shape(28 * 28).unwrap();
    let nn = NNBuilder::new_from_model_file("data/model_9394.bin")
        .unwrap()
        .build()
        .unwrap();
    let output = nn.feedforward(input_1d.view());
    if let Some((idx, percent)) = output
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
    {
        println!("identified as: {}, percent: {:.2}", idx, percent);
    } else {
        println!("The vector is empty");
    }
    // Create a window
    let window = show_image::create_window("image", Default::default())
        .map_err(|e| e.to_string())
        .unwrap();

    // Convert the image to ImageView
    let img_view = ImageView::new(ImageInfo::rgb8(28, 28), image.as_raw());

    // Display the image
    window.set_image("mnist", img_view).unwrap();

    for event in window.event_channel().map_err(|e| e.to_string()).unwrap() {
        if let event::WindowEvent::KeyboardInput(event) = event {
            if !event.is_synthetic
                && event.input.key_code == Some(event::VirtualKeyCode::Escape)
                && event.input.state.is_pressed()
            {
                println!("Escape pressed!");
                break;
            }
        }
    }
}

fn return_item_description_from_number(val: u8) {
    let description = match val {
        0 => "0",
        1 => "1",
        2 => "2",
        3 => "3",
        4 => "4",
        5 => "5",
        6 => "6",
        7 => "7",
        8 => "8",
        9 => "9",
        _ => panic!("An unrecognized label was used..."),
    };
    println!(
        "Based on the '{}' label, this image should be a: {} ",
        val, description
    );
    println!("Hit [ ESC ] to exit...");
}

fn bw_ndarray2_to_rgb_image(arr: Array2<f32>) -> RgbImage {
    assert!(arr.is_standard_layout());

    let (width, height) = (arr.ncols(), arr.ncols());
    let mut img: RgbImage = ImageBuffer::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let val = (arr[[y, x]] * 255.) as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([val, val, val]))
        }
    }
    img
}
