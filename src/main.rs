#![allow(dead_code, unused)]

use gpu_mining::*;

mod chains;
mod server;
mod utils;


fn main() {
    let hash = cuda_sha256("069732");
    println!("{:?}", hash);
    
    let p = gpu_properties();
    println!("{:?}", p);
    crate::server::routers::run();
}
