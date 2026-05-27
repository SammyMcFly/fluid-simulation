/// macros.rs

#[cfg(not(feature = "parallelized_sph"))]
macro_rules! for_each {
    // 1 mutable field
    (
        mut [$m1:expr],
        ref [$($ref_name:ident = $ref_expr:expr),* $(,)?],
        |$id:ident, $a:ident| $body:expr
    ) => {{
        $(let $ref_name = &$ref_expr;)*
        $m1.iter_mut()
            .enumerate()
            .for_each(|($id, $a)| { $body })
    }};

    // 2 mutable fields
    (
        mut [$m1:expr, $m2:expr],
        ref [$($ref_name:ident = $ref_expr:expr),* $(,)?],
        |$id:ident, $a:ident, $b:ident| $body:expr
    ) => {{
        $(let $ref_name = &$ref_expr;)*
        $m1.iter_mut()
            .zip($m2.iter_mut())
            .enumerate()
            .for_each(|($id, ($a, $b))| { $body })
    }};

    // 3 mutable fields
    (
        mut [$m1:expr, $m2:expr, $m3:expr],
        ref [$($ref_name:ident = $ref_expr:expr),* $(,)?],
        |$id:ident, $a:ident, $b:ident, $c:ident| $body:expr
    ) => {{
        $(let $ref_name = &$ref_expr;)*
        $m1.iter_mut()
            .zip($m2.iter_mut())
            .zip($m3.iter_mut())
            .enumerate()
            .for_each(|($id, (($a, $b), $c))| { $body })
    }};

    // 4 mutable fields
    (
        mut [$m1:expr, $m2:expr, $m3:expr, $m4:expr],
        ref [$($ref_name:ident = $ref_expr:expr),* $(,)?],
        |$id:ident, $a:ident, $b:ident, $c:ident, $d:ident| $body:expr
    ) => {{
        $(let $ref_name = &$ref_expr;)*
        $m1.iter_mut()
            .zip($m2.iter_mut())
            .zip($m3.iter_mut())
            .zip($m4.iter_mut())
            .enumerate()
            .for_each(|($id, ((($a, $b), $c), $d))| { $body })
    }};
}

#[cfg(feature = "parallelized_sph")]
macro_rules! for_each {
    // 1 mutable field
    (
        mut [$m1:expr],
        ref [$($ref_name:ident = $ref_expr:expr),* $(,)?],
        |$id:ident, $a:ident| $body:expr
    ) => {{
        $(let $ref_name = &$ref_expr;)*
        $m1.par_iter_mut()
            .enumerate()
            .for_each(|($id, $a)| { $body })
    }};

    // 2 mutable fields
    (
        mut [$m1:expr, $m2:expr],
        ref [$($ref_name:ident = $ref_expr:expr),* $(,)?],
        |$id:ident, $a:ident, $b:ident| $body:expr
    ) => {{
        $(let $ref_name = &$ref_expr;)*
        $m1.par_iter_mut()
            .zip($m2.par_iter_mut())
            .enumerate()
            .for_each(|($id, ($a, $b))| { $body })
    }};

    // 3 mutable fields
    (
        mut [$m1:expr, $m2:expr, $m3:expr],
        ref [$($ref_name:ident = $ref_expr:expr),* $(,)?],
        |$id:ident, $a:ident, $b:ident, $c:ident| $body:expr
    ) => {{
        $(let $ref_name = &$ref_expr;)*
        $m1.par_iter_mut()
            .zip($m2.par_iter_mut())
            .zip($m3.par_iter_mut())
            .enumerate()
            .for_each(|($id, (($a, $b), $c))| { $body })
    }};

    // 4 mutable fields
    (
        mut [$m1:expr, $m2:expr, $m3:expr, $m4:expr],
        ref [$($ref_name:ident = $ref_expr:expr),* $(,)?],
        |$id:ident, $a:ident, $b:ident, $c:ident, $d:ident| $body:expr
    ) => {{
        $(let $ref_name = &$ref_expr;)*
        $m1.par_iter_mut()
            .zip($m2.par_iter_mut())
            .zip($m3.par_iter_mut())
            .zip($m4.par_iter_mut())
            .enumerate()
            .for_each(|($id, ((($a, $b), $c), $d))| { $body })
    }};
}

pub(crate) use for_each;
