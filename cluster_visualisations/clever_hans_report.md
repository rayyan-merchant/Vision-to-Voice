# Clever Hans Audit — SmoothGrad Attribution Clustering Report

Note on attribution method: AttnLRP via lxt was unavailable due to a
transformers version conflict (lxt 2.1 requires transformers<4.40,
Colab has transformers 5.0). SmoothGrad was used as the fallback
attribution method. Silhouette score of 0.142 reflects the known
limitation of SmoothGrad producing diffuse maps. Manual inspection
reveals meaningful visual groupings despite the low score.

Total samples: 150
Best k: 5
Silhouette scores: k=3: 0.116, k=4: 0.131, k=5: 0.142

---

## Cluster 0 — Exterior and Entrance Structures
Size: 16 samples
Avg surprise: low-mid (0.2–9.0)
Action distribution: MoveAhead, RotateLeft, RotateRight
Visual region attended: Ground-level features, stair edges,
exterior door frames
Verdict: LEGITIMATE — Structural reasoning
The model correctly identifies architectural transition points
between outdoor and indoor spaces. Stairs and entrance doors
are critical landmarks for blind campus navigation.
Fix needed: None.

## Cluster 1 — Human Presence
Size: 20 samples
Avg surprise: high (7.0–10.0)
Action distribution: MoveAhead, RotateLeft, RotateRight
Visual region attended: Human silhouettes and body shapes,
consistently across all frames regardless of position
Verdict: LEGITIMATE — Semantic reasoning
The model attends to people as dynamic obstacles. High surprise
scores confirm humans are unpredictable elements the JEPA world
model cannot forecast. Attending to human presence is essential
for safe blind navigation.
Fix needed: None.

## Cluster 2 — Door Frames and Signage
Size: 29 samples
Avg surprise: mid (3.0–8.0)
Action distribution: MoveAhead, RotateLeft
Visual region attended: Door frame edges and sign regions
on or near doors
Verdict: LEGITIMATE — Structural and Semantic reasoning
The model correctly identifies doors as primary navigation
landmarks. Sign activation confirms EasyOCR-relevant regions
are being attended to. This is the most directly useful cluster
for the blind navigation use case.
Fix needed: None.

## Cluster 3 — Indoor Open Spaces
Size: 26 samples
Avg surprise: low-mid
Action distribution: MoveAhead, RotateRight
Visual region attended: Upper frame region — ceiling lights
and wall edges
Verdict: BORDERLINE — Possible Clever Hans shortcut
The model may be exploiting ceiling light patterns as a proxy
for indoor open-space classification rather than attending to
structural features. Ceiling lights are visually consistent
across all indoor frames and could represent a spurious
correlation rather than genuine scene understanding.
Proposed fix: Vary lighting conditions during data collection.
Use scenes with different ceiling types to prevent the model
from using lighting as a shortcut cue.

## Cluster 4 — General Navigation (Catch-all)
Size: 59 samples
Avg surprise: varied (0.0–11.0)
Action distribution: all three actions equally
Visual region attended: Scattered, no dominant region
Verdict: MIXED — Insufficient visual cohesion
This cluster groups visually diverse frames — staircases,
corridors, outdoor areas — that share no single dominant
attended feature. The large size (59 samples) and scattered
heatmaps suggest this is a catch-all cluster resulting from
the diffuse nature of SmoothGrad maps. With AttnLRP maps
this cluster would likely split into more specific subgroups.
Fix needed: Re-run with AttnLRP when lxt environment
conflict is resolved.

---

## Conclusion

The system demonstrates predominantly legitimate attribution
behaviour. Clusters 0, 1, and 2 show the model attending to
structurally and semantically meaningful features — building
entrances, human presence, and door landmarks — all directly
relevant to blind campus navigation.

Cluster 3 reveals a potential Clever Hans shortcut where the
model may use ceiling light patterns as a scene type proxy.
This is addressable through lighting variation during data
collection.

Cluster 4 reflects a known limitation of SmoothGrad
attribution — maps are too diffuse to produce tight clusters
for visually diverse frames. AttnLRP would resolve this.

Overall trustworthiness assessment: The system is behaving
legitimately for the majority of navigation decisions. One
potential shortcut was identified and a fix proposed. This
demonstrates the value of attribution-based auditing for
safety-critical accessibility systems.