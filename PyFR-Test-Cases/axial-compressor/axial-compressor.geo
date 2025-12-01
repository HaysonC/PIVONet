// Axial compressor annulus geometry (parameterized)
// Generates an annular surface (hub -> tip) discretized in azimuth.

// Parameters
hub_radius = 0.05;       // inner radius (m)
tip_radius = 0.15;       // outer radius (m)
ntheta = 128;            // azimuthal divisions (increase for fidelity)
lc = 0.005;              // target mesh size

// Center point
Point(1) = {0, 0, 0, lc};

// Create inner and outer ring points
// Inner points: tags 2..(ntheta+1)
// Outer points: tags (ntheta+2)..(2*ntheta+1)
For i In {0:ntheta-1}
  ang = 2*Pi*i/ntheta;
  x = hub_radius * Cos(ang);
  y = hub_radius * Sin(ang);
  Point(2 + i) = {x, y, 0, lc};
EndFor

For i In {0:ntheta-1}
  ang = 2*Pi*i/ntheta;
  x = tip_radius * Cos(ang);
  y = tip_radius * Sin(ang);
  Point(2 + ntheta + i) = {x, y, 0, lc};
EndFor

// Build circular arcs for inner and outer boundaries using center point (1) as center
// Inner arcs: curve tags 1..ntheta
// Outer arcs: curve tags (ntheta+1)..(2*ntheta)
For i In {0:ntheta-1}
  p1 = 2 + i;
  p2 = 2 + ((i+1) % ntheta);
  Circle(i+1) = {p1, 1, p2};
EndFor

For i In {0:ntheta-1}
  p1 = 2 + ntheta + i;
  p2 = 2 + ntheta + ((i+1) % ntheta);
  Circle(ntheta + i + 1) = {p1, 1, p2};
EndFor

// Line loops
Line Loop(1) = {ntheta+1:2*ntheta};   // outer loop (Gmsh supports range syntax)
Line Loop(2) = {1:ntheta};             // inner loop

// Plane surface with hole (outer loop, inner loop)
Plane Surface(1) = {1, 2};

// Physical groups
Physical Surface(1) = {1};
Physical Line(1) = {1:2*ntheta};

// Mesh options
Mesh.Algorithm = 6; // Frontal-Delaunay for 2D
Mesh.Format = 2;    // Output msh2 (legacy) so PyFR's reader is compatible
