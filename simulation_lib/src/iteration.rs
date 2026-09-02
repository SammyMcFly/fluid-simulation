//! iteration.rs

#[cfg(not(feature = "parallel"))]
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

#[cfg(feature = "parallel")]
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

#[cfg(not(feature = "parallel"))]
macro_rules! for_each_collect {
    // 1 mutable field, mit lokalem Collector
    (
        mut [$m1:expr],
        ref [$($ref_name:ident = $ref_expr:expr),* $(,)?],
        |$id:ident, $a:ident, $local:ident| $body:expr
    ) => {{
        $(let $ref_name = &$ref_expr;)*
        let mut $local = Vec::new();
        $m1.iter_mut()
            .enumerate()
            .for_each(|($id, $a)| { $body });
        $local
    }};
}

#[cfg(feature = "parallel")]
macro_rules! for_each_collect {
    // 1 mutable field, mit lokalem Collector (fold + reduce)
    (
        mut [$m1:expr],
        ref [$($ref_name:ident = $ref_expr:expr),* $(,)?],
        |$id:ident, $a:ident, $local:ident| $body:expr
    ) => {{
        $(let $ref_name = &$ref_expr;)*
        $m1.par_iter_mut()
            .enumerate()
            .fold(Vec::new, |mut $local, ($id, $a)| {
                $body;
                $local
            })
            .reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                a
            })
    }};
}

pub(crate) use for_each;
pub(crate) use for_each_collect;

#[cfg(test)]
mod tests {
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    // ─── for_each: 1 mutable field ───────────────────────────────────────

    #[test]
    fn for_each_one_field_applies_body_to_every_element() {
        let mut values = vec![1, 2, 3, 4];
        for_each!(
            mut [values],
            ref [],
            |_id, v| {
                *v *= 2;
            }
        );
        assert_eq!(values, vec![2, 4, 6, 8]);
    }

    #[test]
    fn for_each_one_field_provides_correct_index_per_element() {
        let mut values = vec![10, 20, 30];
        let mut ids_seen = vec![0usize; values.len()];
        for_each!(
            mut [values, ids_seen],
            ref [],
            |id, _v, id_slot| {
                *id_slot = id;
            }
        );
        assert_eq!(ids_seen, vec![0, 1, 2]);
    }

    #[test]
    fn for_each_on_empty_slice_is_a_noop() {
        let mut values: Vec<i32> = Vec::new();
        for_each!(
            mut [values],
            ref [],
            |_id, v| {
                *v += 1;
            }
        );
        assert!(values.is_empty());
    }

    // ─── for_each: ref bindings ───────────────────────────────────────────

    #[test]
    fn for_each_single_ref_binding_is_accessible_in_body() {
        let mut output = vec![0; 3];
        let multiplier = vec![10, 20, 30];
        for_each!(
            mut [output],
            ref [mult = multiplier],
            |id, out| {
                *out = mult[id];
            }
        );
        assert_eq!(output, vec![10, 20, 30]);
    }

    #[test]
    fn for_each_multiple_ref_bindings_are_all_accessible() {
        let mut output = vec![0; 3];
        let a = vec![1, 2, 3];
        let b = vec![10, 20, 30];
        for_each!(
            mut [output],
            ref [a = a, b = b],
            |id, out| {
                *out = a[id] + b[id];
            }
        );
        assert_eq!(output, vec![11, 22, 33]);
    }

    // ─── for_each: 2 / 3 / 4 mutable fields ────────────────────────────────

    #[test]
    fn for_each_two_fields_updates_both_in_lockstep() {
        let mut a = vec![1, 2, 3];
        let mut b = vec![10, 20, 30];
        for_each!(
            mut [a, b],
            ref [],
            |_id, x, y| {
                *x += 1;
                *y += 1;
            }
        );
        assert_eq!(a, vec![2, 3, 4]);
        assert_eq!(b, vec![11, 21, 31]);
    }

    #[test]
    fn for_each_three_fields_updates_all_in_lockstep() {
        let mut a = vec![1, 2, 3];
        let mut b = vec![10, 20, 30];
        let mut c = vec![100, 200, 300];
        for_each!(
            mut [a, b, c],
            ref [],
            |_id, x, y, z| {
                *x += 1;
                *y += 1;
                *z += 1;
            }
        );
        assert_eq!(a, vec![2, 3, 4]);
        assert_eq!(b, vec![11, 21, 31]);
        assert_eq!(c, vec![101, 201, 301]);
    }

    #[test]
    fn for_each_four_fields_updates_all_in_lockstep() {
        let mut a = vec![1, 2];
        let mut b = vec![10, 20];
        let mut c = vec![100, 200];
        let mut d = vec![1000, 2000];
        for_each!(
            mut [a, b, c, d],
            ref [],
            |_id, w, x, y, z| {
                *w += 1;
                *x += 1;
                *y += 1;
                *z += 1;
            }
        );
        assert_eq!(a, vec![2, 3]);
        assert_eq!(b, vec![11, 21]);
        assert_eq!(c, vec![101, 201]);
        assert_eq!(d, vec![1001, 2001]);
    }

    #[test]
    fn for_each_stops_at_the_shortest_of_several_mismatched_length_fields() {
        // Mirrors real usage: `zip`-based semantics mean a shorter second
        // field silently caps how many elements of the first field are
        // visited at all — documenting this rather than assuming all
        // production call sites always pass equal-length slices.
        let mut a = vec![1, 2, 3, 4];
        let mut b = vec![10, 20]; // shorter
        for_each!(
            mut [a, b],
            ref [],
            |_id, x, y| {
                *x += 100;
                *y += 100;
            }
        );
        assert_eq!(a, vec![101, 102, 3, 4]);
        assert_eq!(b, vec![110, 120]);
    }

    // ─── for_each_collect ───────────────────────────────────────────────────

    #[test]
    fn for_each_collect_gathers_pushed_values() {
        let mut values = vec![1, 2, 3, 4, 5];
        let mut result = for_each_collect!(
            mut [values],
            ref [],
            |_id, v, local| {
                if *v % 2 == 0 {
                    local.push(*v);
                }
            }
        );
        // Order is not guaranteed under the `parallel` feature (fold+reduce
        // may combine chunks in any order), so sort before comparing.
        result.sort();
        assert_eq!(result, vec![2, 4]);
    }

    #[test]
    fn for_each_collect_can_also_mutate_the_iterated_field() {
        let mut values = vec![1, 2, 3];
        let mut doubled_evens = for_each_collect!(
            mut [values],
            ref [],
            |_id, v, local| {
                *v *= 10;
                if *v % 20 == 0 {
                    local.push(*v);
                }
            }
        );
        assert_eq!(values, vec![10, 20, 30]);
        doubled_evens.sort();
        assert_eq!(doubled_evens, vec![20]);
    }

    #[test]
    fn for_each_collect_ref_binding_is_accessible_in_body() {
        let mut values = vec![0, 1, 2, 3];
        let threshold = 2usize;
        let mut ids_above_threshold = for_each_collect!(
            mut [values],
            ref [t = threshold],
            |id, v, local| {
                *v += 1;
                if id >= *t {
                    local.push(id);
                }
            }
        );
        ids_above_threshold.sort();
        assert_eq!(ids_above_threshold, vec![2, 3]);
        assert_eq!(values, vec![1, 2, 3, 4]);
    }

    #[test]
    fn for_each_collect_visits_every_element_exactly_once_with_correct_id() {
        let mut values = vec![100, 200, 300, 400];
        let mut ids = for_each_collect!(
            mut [values],
            ref [],
            |id, _v, local| {
                local.push(id);
            }
        );
        ids.sort();
        assert_eq!(ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn for_each_collect_on_empty_slice_yields_empty_vec() {
        let mut values: Vec<i32> = Vec::new();
        let result = for_each_collect!(
            mut [values],
            ref [],
            |_id, v, local| {
                local.push(*v);
            }
        );
        assert!(result.is_empty());
    }

    #[test]
    fn for_each_collect_can_push_multiple_items_per_element() {
        let mut values = vec![1, 2, 3];
        let mut result = for_each_collect!(
            mut [values],
            ref [],
            |_id, v, local| {
                local.push(*v);
                local.push(*v * 10);
            }
        );
        result.sort();
        assert_eq!(result, vec![1, 2, 3, 10, 20, 30]);
    }
}
