use base64::{Engine as _, engine::general_purpose};

fn main() {
    // Your base64 string (truncated for brevity)
    let base64_data = "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAABfUlEQVRIS8WWoU4DQRCGu6GEhCBQaFAIJKpP0FSR1FSDIWB4ARwvgANEi69pZcMTIKoRGLAoBAICJMd3F+6yuezOTMOxbbJJ25n7v/lnd6d1rcQvl5jXMgGzLJtS2EGguFvn3NEiRatAYJkmCFTVKTXERB8WEvXi78TXtcLyeBSI2AbxtyJJcFBCrS41h8cI3UiVA/wmvtII0NIigBfknbNOgV5pz5g3OyYE8IPYGmsH4HMKYHGKk7QUd2ewLlknAK81d+IprT+M+CvfbYZEre5MQEBbJL5o1Vuh6qERJk0fyIR4PtqG1SRRpo52Dz8RWq2566B5H2h5NQIlt6pDrZV+HLdzPu+zekBnwf1eRNCSq426Rh3mBS0LeEdLu//eUs1d9B7+PvhAlXuWfStz/go0z0fL3lX3NOSASp/4frtogeHvA/lfpLYt+dIvvv9fZhfuY6Q404UXHXp7MuL9oQfKxSesfh1u6UT00EhjK+ByDGxgPVyNX3wNnBz4A9bykh1vbHb/AAAAAElFTkSuQmCC";

    // Remove the prefix `data:image/png;base64,` if present
    let base64_data = base64_data.trim_start_matches("data:image/png;base64,");

    // Decode the base64 string to bytes
    let decoded_bytes = general_purpose::STANDARD.decode(base64_data).unwrap();

    // Load the image from the decoded bytes
    let img = image::load_from_memory(&decoded_bytes).expect("Failed to decode image");

    // You can now manipulate the image object as needed
    // For example, saving it to a file
    img.save("data/three.png").expect("Failed to save image");
}
