# ConveyorBench V1 procedural object library

This directory contains the canonical, machine-readable registry for the first
eight conveyor parts. The geometry is constructed at runtime from project-local
primitive recipes; it does not download, reference, or payload external files.

The initial library deliberately varies grasp geometry instead of recoloring one
cube:

- compact cuboid;
- long bar;
- upright bushing;
- horizontal shaft;
- hexagonal blank;
- stepped flange;
- L bracket;
- gear-like disk proxy.

The registry owns the coarse six-`seen`/two-`unseen` split. ConveyorBench tasking
then freezes three mutually exclusive curriculum partitions:

- `train`: red block, blue bar, yellow bushing and green shaft;
- `val`: silver hex and orange flange;
- `unseen`: purple bracket and cyan gear.

Every entry freezes language aliases, mass, friction, optional calibrated rigid
body damping, stable poses and a primary parallel-jaw grasp affordance. The
upright yellow bushing uses angular damping to model the rolling resistance
that prevents a polymer cylinder from rolling forever after a tray drop. All
dimensions use metres and all masses use kilograms. Runtime manifests preserve
both the coarse registry split and the three-way curriculum split; exporters
may not reassign either.
