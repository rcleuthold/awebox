import awebox.mdl.aero.induction_dir.vortex_dir.vortex as vortex
import awebox.mdl.aero.geometry_dir.geometry as geometry
import awebox.mdl.aero.tether_dir.tether as tether
import awebox.mdl.aero.induction_dir.actuator_dir.actuator as actuator

def test_actuator_components():
    actuator.independent_test()
    return None

def test_vortex_components():
    vortex.test()
    return None

def test_geometry_components():
    geometry.test()
    return None

def test_tether_drag():
    tether.test()
    return None

if __name__ == "__main__":
    test_actuator_components()
    test_vortex_components()
    test_geometry_components()
    test_tether_drag()