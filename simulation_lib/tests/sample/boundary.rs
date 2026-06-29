use core::f64;
use simulation_lib::fluid::{Boundary3D, SerBoundary3D, Len, Expandable, Positional};
use nalgebra::Vector3;

fn v(x: f64, y: f64, z: f64) -> Vector3<f64> {
    Vector3::new(x, y, z)
}

    // ─── Boundary3D: Len trait ──────────────────────────────────────────

    #[test]
    fn boundary_default_is_empty() {
        let boundary = Boundary3D::default();
        assert_eq!(boundary.len(), 0);
        assert!(boundary.is_empty());
    }

    // ─── Boundary3D: Expandable trait ───────────────────────────────────

    #[test]
    fn boundary_push() {
        let mut boundary = Boundary3D::default();
        boundary.push(v(1.0, 2.0, 3.0), v(0.1, 0.2, 0.3), 0.5);

        assert_eq!(boundary.len(), 1);
        assert_eq!(boundary.position[0], v(1.0, 2.0, 3.0));
        assert_eq!(*boundary.vel_now(0), v(0.1, 0.2, 0.3));
        assert_eq!(*boundary.volume(0), 0.5);
    }

    #[test]
    fn boundary_push_multiple() {
        let mut boundary = Boundary3D::default();
        boundary.push(v(1.0, 0.0, 0.0), Vector3::zeros(), 0.1);
        boundary.push(v(2.0, 0.0, 0.0), Vector3::zeros(), 0.2);
        boundary.push(v(3.0, 0.0, 0.0), Vector3::zeros(), 0.3);

        assert_eq!(boundary.len(), 3);
    }

    #[test]
    fn boundary_extend() {
        let mut b_a = Boundary3D::default();
        b_a.push(v(1.0, 0.0, 0.0), Vector3::zeros(), 0.1);

        let mut b_b = Boundary3D::default();
        b_b.push(v(2.0, 0.0, 0.0), Vector3::zeros(), 0.2);
        b_b.push(v(3.0, 0.0, 0.0), Vector3::zeros(), 0.3);

        b_a.extend(b_b);

        assert_eq!(b_a.len(), 3);
        assert_eq!(b_a.position[1], v(2.0, 0.0, 0.0));
        assert_eq!(b_a.position[2], v(3.0, 0.0, 0.0));
    }

    // ─── Boundary3D: Positional trait ───────────────────────────────────

    #[test]
    fn boundary_pos_now() {
        let mut boundary = Boundary3D::default();
        boundary.push(v(5.0, 6.0, 7.0), Vector3::zeros(), 1.0);

        assert_eq!(*boundary.pos_now(0), v(5.0, 6.0, 7.0));
    }

    #[test]
    fn boundary_pos_now_range() {
        let mut boundary = Boundary3D::default();
        boundary.push(v(1.0, 0.0, 0.0), Vector3::zeros(), 1.0);
        boundary.push(v(2.0, 0.0, 0.0), Vector3::zeros(), 1.0);

        let slice = boundary.pos_now(0..2);
        assert_eq!(slice.len(), 2);
    }

    // ─── Boundary3D: volume ─────────────────────────────────────────────

    #[test]
    fn boundary_set_volume() {
        let mut boundary = Boundary3D::default();
        boundary.push(v(0.0, 0.0, 0.0), Vector3::zeros(), 0.0);

        boundary.set_volume(0, f64::consts::PI);
        assert_eq!(*boundary.volume(0), f64::consts::PI);
    }

    #[test]
    fn boundary_volume_range() {
        let mut boundary = Boundary3D::default();
        boundary.push(v(0.0, 0.0, 0.0), Vector3::zeros(), 1.0);
        boundary.push(v(0.0, 0.0, 0.0), Vector3::zeros(), 2.0);
        boundary.push(v(0.0, 0.0, 0.0), Vector3::zeros(), 3.0);

        let slice = boundary.volume(0..3);
        assert_eq!(slice, &[1.0, 2.0, 3.0]);
    }

    // ─── Boundary3D: vel_now ────────────────────────────────────────────

    #[test]
    fn boundary_vel_now_range() {
        let mut boundary = Boundary3D::default();
        boundary.push(v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), 0.0);
        boundary.push(v(0.0, 0.0, 0.0), v(2.0, 0.0, 0.0), 0.0);

        let slice = boundary.vel_now(0..2);
        assert_eq!(slice[0], v(1.0, 0.0, 0.0));
        assert_eq!(slice[1], v(2.0, 0.0, 0.0));
    }

    // ─── SerBoundary3D <-> Boundary3D conversions ───────────────────────

    #[test]
    fn boundary_from_ser_boundary() {
        let ser = SerBoundary3D {
            position: vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            velocity: vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        };

        let boundary: Boundary3D = ser.into();

        assert_eq!(boundary.len(), 2);
        assert_eq!(boundary.position[0], v(1.0, 2.0, 3.0));
        assert_eq!(boundary.position[1], v(4.0, 5.0, 6.0));
        assert_eq!(*boundary.vel_now(0), v(0.1, 0.2, 0.3));
        assert_eq!(*boundary.volume(0), 0.0); // initialized to 0
    }

    #[test]
    fn ser_boundary_from_boundary() {
        let mut boundary = Boundary3D::default();
        boundary.push(v(1.0, 2.0, 3.0), v(0.1, 0.2, 0.3), 1.0);

        let ser: SerBoundary3D = boundary.into();

        assert_eq!(ser.position, vec![[1.0, 2.0, 3.0]]);
        assert_eq!(ser.velocity, vec![[0.1, 0.2, 0.3]]);
    }

    #[test]
    fn boundary_roundtrip_conversion() {
        let mut original = Boundary3D::default();
        original.push(v(1.0, 2.0, 3.0), v(0.5, 0.5, 0.5), 0.0);
        original.push(v(4.0, 5.0, 6.0), v(1.0, 1.0, 1.0), 0.0);

        let ser: SerBoundary3D = original.clone().into();
        let restored: Boundary3D = ser.into();

        assert_eq!(restored.len(), original.len());
        for i in 0..original.len() {
            assert_eq!(restored.position[i], original.position[i]);
            assert_eq!(*restored.vel_now(i), *original.vel_now(i));
        }
    }
