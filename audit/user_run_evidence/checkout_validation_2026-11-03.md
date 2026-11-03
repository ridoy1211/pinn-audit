# Validation of the Git checkout environment

Reviewed 3 November 2026, following merge of PR 1 at d1de65e.
Evidence: the user's CMD output supplied in chat; no independent repetition of
this exact run was performed by the assistant.

The printed interpreter is the .venv inside the canonical Git checkout, and
pinnaudit is imported from that checkout's src/pinnaudit/__init__.py.

Command:

    python -m pytest --basetemp work_test_temp_audit_checkout_1

Reported result:

    180 passed in 37.98s

The working tree was clean on main immediately before preparation of the separate
sparse-inverse-diagnostic branch. This establishes the intended code/environment
location for the next reproduction. It is a software-validation record, not a new
physical experiment or evidence of comparative PINN performance.
