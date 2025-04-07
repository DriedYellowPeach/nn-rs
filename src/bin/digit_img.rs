use nn_rs::nn::NNBuilder;

fn main() {
    // Load the PNG image from the file
    let img = image::open("data/three.png").expect("Failed to open image");

    // Convert the image to grayscale
    let grayscale_img = img.to_luma8(); // Converts to grayscale (Luma8 means an 8-bit grayscale image)
    let gray_values = grayscale_img.pixels().map(|p| p[0]).collect::<Vec<u8>>();

    for (i, &value) in gray_values.iter().enumerate() {
        if value < 128 {
            print!(".");
        } else {
            print!("x");
        }
        if (i + 1) % grayscale_img.width() as usize == 0 {
            println!();
        }
    }

    let nn = NNBuilder::new_from_model_file("data/model_9394.bin")
        .unwrap()
        .build()
        .unwrap();

    let ret = nn_rs::recognize_digit(&nn, &gray_values);
    println!("Recognized digit: {}, confidence: {}", ret.0, ret.1);
}
