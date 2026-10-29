# Project scope

madmom-plus is a general music-analysis library. It keeps madmom's purpose, features, public Python API, processors, command-line programs, and model compatibility.

A change belongs in madmom-plus when its input is audio and its output is a musical property, such as activations, beats, downbeats, tempo, or tracker state. A madmom user must be able to use the change without knowledge of a particular downstream application. Public data and tests in madmom's style must justify the change.

Defaults change only with public evidence and without breaking madmom's tests. New behavior is additive or opt-in unless it fixes a bug. Existing tests and their numerical tolerances must not be weakened to make a change pass.

Application policy belongs downstream. This includes presence and music-detection decisions, pulse timing contracts, clocks, transports such as OSC, JSON, and UDP, application command-line services, device packaging, and private evaluation systems. These features do not belong in this library.

The dependency goes one way: applications depend on a pinned madmom-plus revision. madmom-plus never depends on a downstream application.

madmom-plus exposes a raw tracker reliability signal. Downstream applications own the policy that decides whether to use or publish that signal. A maximum DBN posterior is not a calibrated music-presence probability.
