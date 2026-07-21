import sys; sys.path.insert(0, "src")
from amoe.testing.invariants import run_all
run_all()
from amoe.testing.diffusion_invariants import run_all as run_diffusion
run_diffusion()
