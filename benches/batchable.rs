use candle_core::{Device, Tensor};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use oven::traits::*;
use std::hint::black_box;

fn make_tensor(device: &Device) -> Tensor {
    let flat = [1i64; 10];
    Tensor::from_slice(&flat, (2, 5), device).unwrap()
}

fn bench_push(c: &mut Criterion) {
    let device = Device::Cpu;

    c.bench_function("buffer push", |b| {
        b.iter_batched(
            // setup: 측정에서 제외됨
            || (Vec::with_capacity(10_000), make_tensor(&device)),
            // routine: 이 부분만 측정됨
            |(mut buffer, tensor)| {
                buffer.push(tensor);
                black_box(buffer);
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_batch(c: &mut Criterion) {
    let device = Device::Cpu;
    // batch 대상 데이터는 미리 한 번만 만들어둠
    let source: Vec<Tensor> = (0..10_000).map(|_| make_tensor(&device)).collect();

    c.bench_function("batch", |b| {
        b.iter_batched(
            || source.clone(), // Tensor clone은 얕은 복사(참조 카운트)라 저렴함
            |buffer| black_box(Tensor::batch(buffer, &device).unwrap()),
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, bench_push, bench_batch);
criterion_main!(benches);