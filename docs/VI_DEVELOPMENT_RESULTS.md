# Vietnamese development results

Measured on 2026-11-22 using the local Chrome SODA Vietnamese pack and three user-provided development recordings.

The main engineering discovery was that feeding long audio too quickly caused SODA to drop entire sections. Pacing the stream at a safe rate restored the missing content. The recordings were also unusually quiet, so an experiment used additional gain only for those files; that gain is not a recommended microphone default.

After normalizing equivalent number spellings, the three development samples contained 12 errors across 210 whitespace-delimited reference tokens (5.71%). These clips were used during configuration work, so they are a development set rather than an independent test set.

A Palace-derived contextual-bias candidate did not improve the aggregate result and was not promoted. An experimental change to a copy of 3,745 output-bias parameters reduced text-training loss but did not reduce recognition errors on these three clips, so it was not promoted either.

The current normal path therefore uses the original SODA weights plus optional user-controlled Vietnamese context biasing and deterministic post-correction.
