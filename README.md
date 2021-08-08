# grid-geom-util
A geometric utility library made mostly for grid-based games, such as roguelikes.

I'm using this in a personal videogame project, but I'll hopefully make it good enough for general use someday.
Contributions and github issues are welcome.

There isn't any crates.io crate yet released, or any versioning.
This library is still under heavy development and may change a lot.
I'll try to publish it properly once I feel it's mature enough.

## Usage
The main types of this library are `Point`, `Line`, and `Rect`.
All types can be converted to and from tuples and arrays with the standard `Into` and `From` traits.

`Point` and `Line` are simple point and line-segment 2D primitives.
`Line` consists of a start position x1 and y1, and an end position x2 and y2.

### Rect

`Rect` is an axis-aligned rectangle consisting of a minimum-coordinate position x1 and y1, and a maximum-coordinate position x2 and y2.
These are *inclusive*, meaning that if they're the same then it's a rectangle with the size 1x1, not 0x0.

This is taken into account by the `width` and `height` methods, which return `x2-x1+1` and `y2-y1+1` respectively.
If you want the difference in coordinates, ie. just `x2-x1` or `y2-y1`, use the `diff_x` or `diff_y` method.

Methods of `Rect` also assume that the maximum-coordinates are greater or equal to the minimum coordinates.
If this precondition is broken then any method may panic, return nonsensible results, or do undefined behaviour.

To avoid this `Rect` is marked as `#[non_exhaustive]` to stop anyone to create them manually (like this: `Rect { x1, y1, x2, y2 }`)
and the only way to create a possibly malformed rectangle is with the unsafe `new_unchecked` constructor.


### Type Parameters
All types are generic over the choice of the coordinate type.

Most of the methods however expect them to implement the std traits `Clone` and `PartialOrd`, and the `num-traits` trait `Num`.
Additionally many methods don't really make much sense with non-integer coordinates (this is meant for grid-based games and such),
so you should only use them sparingly, like during conversions and such.

### Display
All types implement the std trait `Display`.
All implementations simply return the coordinate-values delimited by a comma and space, eg. `"1, 5"` or `"2, 3, 5, 5"`.

### (De)serialization
All the types implement serde (de)serializations.
For maximum portability and simpleness they all serialize into and deserialize from regular arrays, ie. `Point<T>` becomes `[T; 2]`, `Line<T>` and `Rect<T>` become `[T; 4]`.

When deserializing into `Rect<T>` the `T` has to be `PartialOrd` in order to make sure it can't be a malformed rectangle.

## License
Licensed under either the <a href="LICENSE-APACHE">Apache License, Version 2.0</a>, or the <a href="LICENSE-MIT">MIT License</a>

### Contribution
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
