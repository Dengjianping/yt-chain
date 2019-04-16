#![allow(dead_code, unused)]

mod chains;
mod server;
mod utils;


fn main() {
    crate::server::routers::run();
}
