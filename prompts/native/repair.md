Repair the wholly invalid Native v3 batch using only the supplied bounded
diagnostics. Reuse exactly the original call ID, slot IDs, frozen parent
snapshot, and requested batch size. Do not create replacement slots.

Return one complete program-batch JSON object. Each `program_json_raw` value
must contain a declarative Native v3 AST, never Python or prose.
