use crate::{CardDir, Rect, max, min};

use num::Float;
use num::{Integer, Num, Signed};
use serde::{Deserialize, Serialize};

use std::fmt;
use std::iter::FusedIterator;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A generic 2D point.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(
    bound(serialize = "T: Clone + Serialize"),
    into = "[T; 2]",
    from = "[T; 2]"
)]
pub struct Point<T> {
    pub x: T,
    pub y: T,
}

impl<T> Point<T> {
    /// Creates a new point.
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    /// Converts self into a tuple.
    ///
    /// This is for when the type can't be inferred when using just [`Into::into`].
    pub fn into_tuple(self) -> (T, T) {
        self.into()
    }

    /// Converts self into an array.
    ///
    /// This is for when the type can't be inferred when using just [`Into::into`].
    pub fn into_array(self) -> [T; 2] {
        self.into()
    }

    /// Maps the same function to the both of the coordinates and returns them as a new point.
    ///
    /// The function will be applied to the x-coordinate first.
    #[must_use]
    pub fn map<F, U>(self, mut f: F) -> Point<U>
    where
        F: FnMut(T) -> U,
    {
        Point {
            x: f(self.x),
            y: f(self.y),
        }
    }

    /// Maps a function to the x coordinate and returns it as a new point.
    #[must_use]
    pub fn map_x<F>(self, mut f: F) -> Point<T>
    where
        F: FnMut(T) -> T,
    {
        Point {
            x: f(self.x),
            y: self.y,
        }
    }

    /// Maps a function to the y coordinate and returns it as a new point.
    #[must_use]
    pub fn map_y<F>(self, mut f: F) -> Point<T>
    where
        F: FnMut(T) -> T,
    {
        Point {
            x: self.x,
            y: f(self.y),
        }
    }
}

impl<T: Clone> Point<T> {
    /// Checks if both values fulfill the given predicate function
    ///
    /// The predicate is short-circuiting and applied to the x-coordinate first.
    #[must_use]
    pub fn all<F>(&self, mut predicate: F) -> bool
    where
        F: FnMut(&T) -> bool,
    {
        predicate(&self.x) && predicate(&self.y)
    }

    /// Checks if either value fulfills the given predicate function
    ///
    /// The predicate is short-circuiting and applied to the x-coordinate first.
    #[must_use]
    pub fn any<F>(&self, mut predicate: F) -> bool
    where
        F: FnMut(&T) -> bool,
    {
        predicate(&self.x) || predicate(&self.y)
    }

    /// Returns the coordinate which is smaller.
    pub fn min_coord(&self) -> T
    where
        T: PartialOrd,
    {
        min(self.x.clone(), self.y.clone())
    }

    /// Returns the coordinate which is greater.
    pub fn max_coord(&self) -> T
    where
        T: PartialOrd,
    {
        max(self.x.clone(), self.y.clone())
    }

    /// Applies the absolute value function for x and y.
    pub fn abs(&self) -> Self
    where
        T: Signed,
    {
        Point::new(self.x.clone().abs(), self.y.clone().abs())
    }

    /// Applies the absolute value function for x and y.
    ///
    /// Does not use the standard `abs` implementation.
    /// Can be used for unsigned values as well, although it has no effect.
    pub fn abs_generic(&self) -> Self
    where
        T: Num + PartialOrd,
    {
        let abs = |x: T| if x < T::zero() { T::zero() - x } else { x };
        Point::new(abs(self.x.clone()), abs(self.y.clone()))
    }

    /// Applies the given function to the coordinates of two points.
    ///
    /// The function is applied to the x-coordinate first.
    pub fn op<U, O, F>(&self, other: &Point<U>, mut operator: F) -> Point<O>
    where
        U: Clone,
        F: FnMut(&T, &U) -> O,
    {
        Point::new(operator(&self.x, &other.x), operator(&self.y, &other.y))
    }

    /// Applies the given function to the coordinates of many points.
    ///
    /// The function is applied to the x-coordinates first.
    pub fn op_many<I, O, F>(arguments: I, mut operator: F) -> Point<O>
    where
        T: Copy,
        I: IntoIterator<Item = Point<T>>,
        F: FnMut(&[T]) -> O,
    {
        let (x_inputs, y_inputs): (Vec<T>, Vec<T>) =
            arguments.into_iter().map(Point::into_tuple).unzip();
        Point::new(operator(&x_inputs[..]), operator(&y_inputs[..]))
    }

    /// Applies the given predicate between the coordinates of two points and returns if both filled it
    ///
    /// The function is applied to the x-coordinate first.
    /// Equivalent to `self.op(other, predicate).all(|x| *x)`.
    pub fn op_all<U, F>(&self, other: &Point<U>, predicate: F) -> bool
    where
        U: Clone,
        F: FnMut(&T, &U) -> bool,
    {
        self.op(other, predicate).all(|x| *x)
    }

    /// Applies the given predicate to the coordinates of many points, and returns if both filled it.
    ///
    /// The function is applied to the x-coordinates first.
    /// Equivalent to `Self::op_many(arguments, operator).all(|x| *x)`.
    pub fn op_all_many<I, F>(arguments: I, operator: F) -> bool
    where
        T: Copy,
        I: IntoIterator<Item = Point<T>>,
        F: FnMut(&[T]) -> bool,
    {
        Self::op_many(arguments, operator).all(|x| *x)
    }

    /// Applies the given predicate between the coordinates of two points and returns if either one filled it
    ///
    /// The function is applied to the x-coordinate first.
    /// Equivalent to `self.op(other, predicate).any(|x| *x)`.
    pub fn op_any<U, F>(&self, other: &Point<U>, predicate: F) -> bool
    where
        U: Clone,
        F: FnMut(&T, &U) -> bool,
    {
        self.op(other, predicate).any(|x| *x)
    }

    /// Applies the given predicate to the coordinates of many points, and returns if either one filled it.
    ///
    /// The function is applied to the x-coordinates first.
    /// Equivalent to `Self::op_many(arguments, operator).any(|x| *x)`.
    pub fn op_any_many<I, F>(arguments: I, operator: F) -> bool
    where
        T: Copy,
        I: IntoIterator<Item = Point<T>>,
        F: FnMut(&[T]) -> bool,
    {
        Self::op_many(arguments, operator).any(|x| *x)
    }

    /// Returns the sum of x and y points.
    pub fn sum(&self) -> T
    where
        T: Num,
    {
        self.x.clone() + self.y.clone()
    }

    /// Returns the point's distance from the origin in the usual euclid space.
    pub fn distance_euclid(&self) -> T
    where
        T: Float,
    {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }

    /// Returns the [taxicab/manhattan distance](https://en.wikipedia.org/wiki/Taxicab_geometry) from the origin.
    ///
    /// Equivalent to `point.abs_generic().sum()`
    pub fn distance_taxi(&self) -> T
    where
        T: Num + PartialOrd,
    {
        self.abs_generic().sum()
    }

    /// Returns the [Chebyshev/king's move distance](https://en.wikipedia.org/wiki/Chebyshev_distance) from the origin.
    ///
    /// Equivalent to `point.abs_generic().max_coord()`.
    pub fn distance_king(&self) -> T
    where
        T: Num + PartialOrd,
    {
        self.abs_generic().max_coord()
    }

    /// Returns the dot product with the given point/vector.
    ///
    /// Equivalent to `(a * b).sum()`.
    pub fn dot(&self, other: &Self) -> T
    where
        T: Num,
    {
        (self * other).sum()
    }

    /// Returns a point with the coordinates reversed `(y, x)`.
    pub fn rev(&self) -> Self {
        Point::new(self.y.clone(), self.x.clone())
    }

    /// Returns the cardinal direction that this point/vector points towards the most.
    ///
    /// The values of coordinates is assumed to grow south-east, ie. x grows towards east and y grows towards south.
    ///
    /// If the length is 0, returns [`CardDir::North`].
    pub fn cardinal(&self) -> CardDir
    where
        T: Signed + PartialOrd,
    {
        if self.x.abs() > self.y.abs() {
            if self.x.is_positive() {
                CardDir::East
            } else {
                CardDir::West
            }
        } else if self.y.is_positive() {
            CardDir::South
        } else {
            CardDir::North
        }
    }

    /// Returns true if this point is inside the given rectangle.
    pub fn is_inside(&self, rect: &Rect<T>) -> bool
    where
        T: PartialOrd,
    {
        self.x >= rect.x1 && self.x <= rect.x2 && self.y >= rect.y1 && self.y <= rect.y2
    }

    /// Moves this point to inside the given rectangle.
    pub fn to_inside(&self, rect: &Rect<T>) -> Point<T>
    where
        T: PartialOrd,
    {
        let Rect { x1, y1, x2, y2 } = rect.clone();
        Point {
            x: num::clamp(self.x.clone(), x1, x2),
            y: num::clamp(self.y.clone(), y1, y2),
        }
    }

    /// Snaps this point to the nearest multiple of the given increment.
    ///
    /// Halfway cases are rounded down towards negative infinity.
    ///
    /// The increment's sign doesn't affect the result.
    ///
    /// # Panics
    /// Panics if either of the increment's values are 0.
    pub fn snap(&self, snap_increment: &Point<T>) -> Self
    where
        T: Integer,
    {
        self.clone()
            .op(&snap_increment.abs_generic(), |x, increment| {
                let prev = x.prev_multiple_of(increment);
                let next = x.next_multiple_of(increment);
                if x.clone() - prev.clone() <= next.clone() - x.clone() {
                    prev
                } else {
                    next
                }
            })
    }
}

impl<T: Clone + Num> Point<T> {
    /// Returns an iterator over the [Von Neumann neighborhood](https://en.wikipedia.org/wiki/Von_Neumann_neighborhood)
    /// of this point.
    pub fn neighbors_neumann_iter(
        &self,
    ) -> impl Iterator<Item = Self> + FusedIterator + ExactSizeIterator + DoubleEndedIterator + use<T>
    {
        let one = T::one;
        let Point { x, y } = self;
        let x = || x.clone();
        let y = || y.clone();

        [
            [x() + one(), y()],
            [x(), y() + one()],
            [x() - one(), y()],
            [x(), y() - one()],
        ]
        .map(Into::into)
        .into_iter()
    }

    /// Returns an iterator over the [Moore neighborhood](https://en.wikipedia.org/wiki/Moore_neighborhood)
    /// of this point.
    ///
    /// The 4 orthogonal (Von Neumann) neighbors are returned first.
    pub fn neighbors_moore_iter(
        &self,
    ) -> impl Iterator<Item = Self> + FusedIterator + ExactSizeIterator + DoubleEndedIterator + use<T>
    {
        let one = T::one;
        let Point { x, y } = self;
        let x = || x.clone();
        let y = || y.clone();

        [
            [x() + one(), y()],
            [x(), y() + one()],
            [x() - one(), y()],
            [x(), y() - one()],
            [x() + one(), y() + one()],
            [x() + one(), y() - one()],
            [x() - one(), y() + one()],
            [x() - one(), y() - one()],
        ]
        .map(Into::into)
        .into_iter()
    }

    /// Returns an iterator over the diagonal neighbors of this point.
    pub fn neighbors_diagonal_iter(
        &self,
    ) -> impl Iterator<Item = Self> + FusedIterator + ExactSizeIterator + DoubleEndedIterator + use<T>
    {
        let one = T::one;
        let Point { x, y } = self;
        let x = || x.clone();
        let y = || y.clone();

        [
            [x() + one(), y() + one()],
            [x() + one(), y() - one()],
            [x() - one(), y() + one()],
            [x() - one(), y() - one()],
        ]
        .map(Into::into)
        .into_iter()
    }
}

impl<T: fmt::Display> fmt::Display for Point<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, {}", self.x, self.y)
    }
}

impl<T> From<[T; 2]> for Point<T> {
    fn from([x, y]: [T; 2]) -> Self {
        Point { x, y }
    }
}

impl<T> From<(T, T)> for Point<T> {
    fn from((x, y): (T, T)) -> Self {
        Point { x, y }
    }
}

impl<T: Clone> From<&[T; 2]> for Point<T> {
    fn from(x: &[T; 2]) -> Self {
        let [x, y] = x.clone();
        Point { x, y }
    }
}

impl<T: Clone> From<&(T, T)> for Point<T> {
    fn from(x: &(T, T)) -> Self {
        let (x, y) = x.clone();
        Point { x, y }
    }
}

impl<T> From<Point<T>> for [T; 2] {
    fn from(Point { x, y }: Point<T>) -> Self {
        [x, y]
    }
}

impl<T> From<Point<T>> for (T, T) {
    fn from(Point { x, y }: Point<T>) -> Self {
        (x, y)
    }
}

macro_rules! impl_ops {
    ($($trait:tt, $fun:ident, $trait_assign:tt, $fun_assign:ident;)*) => {$(
        impl<T: $trait> $trait for Point<T> {
            type Output = Point<<T as $trait>::Output>;
            fn $fun(self, rhs: Point<T>) -> Self::Output {
                Point::new(
                    <T as $trait>::$fun(self.x, rhs.x),
                    <T as $trait>::$fun(self.y, rhs.y),
                )
            }
        }

        impl<T: $trait + Clone> $trait for &Point<T> {
            type Output = Point<<T as $trait>::Output>;
            fn $fun(self, rhs: &Point<T>) -> Self::Output {
                Point::new(
                    <T as $trait>::$fun(self.x.clone(), rhs.x.clone()),
                    <T as $trait>::$fun(self.y.clone(), rhs.y.clone()),
                )
            }
        }

        impl<T: $trait + Clone> $trait::<&Point<T>> for Point<T> {
            type Output = Point<<T as $trait>::Output>;
            fn $fun(self, rhs: &Point<T>) -> Self::Output {
                Point::new(
                    <T as $trait>::$fun(self.x, rhs.x.clone()),
                    <T as $trait>::$fun(self.y, rhs.y.clone()),
                )
            }
        }

        impl<T: $trait + Clone> $trait::<Point<T>> for &Point<T> {
            type Output = Point<<T as $trait>::Output>;
            fn $fun(self, rhs: Point<T>) -> Self::Output {
                Point::new(
                    <T as $trait>::$fun(self.x.clone(), rhs.x),
                    <T as $trait>::$fun(self.y.clone(), rhs.y),
                )
            }
        }

        impl<T: $trait_assign> $trait_assign for Point<T> {
            fn $fun_assign(&mut self, rhs: Point<T>) {
                <T as $trait_assign>::$fun_assign(&mut self.x, rhs.x);
                <T as $trait_assign>::$fun_assign(&mut self.y, rhs.y);
            }
        }

        impl<T: $trait_assign + Clone> $trait_assign::<&Point<T>> for Point<T> {
            fn $fun_assign(&mut self, rhs: &Point<T>) {
                <T as $trait_assign>::$fun_assign(&mut self.x, rhs.x.clone());
                <T as $trait_assign>::$fun_assign(&mut self.y, rhs.y.clone());
            }
        }
    )*};
}

impl_ops!(
    Add, add, AddAssign, add_assign;
    Sub, sub, SubAssign, sub_assign;
    Mul, mul, MulAssign, mul_assign;
    Div, div, DivAssign, div_assign;
    Rem, rem, RemAssign, rem_assign;
);

impl<T: Neg> Neg for Point<T> {
    type Output = Point<T::Output>;
    fn neg(self) -> Self::Output {
        Point::new(-self.x, -self.y)
    }
}
