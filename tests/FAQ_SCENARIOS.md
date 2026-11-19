# FAQ scenario review

All questions use the approved synthetic corpus. Expectations stay in `tests/faq_scenarios.json`; they are never indexed. The update case uses a hypothetical revision only in a temporary collection.

| ID | Question | HTTP / status | Expected fact or reason | Title / page |
| --- | --- | --- | --- | --- |
| statements-download | How can I download electronic statements? | 200 / answered | Download through the secure banking document centre. | Synthetic Statement Policy / 1 |
| statements-download-paraphrase | Where do I get a PDF copy of an electronic statement? | 200 / answered | Download through the secure banking document centre. | Synthetic Statement Policy / 1 |
| statements-frequency | How often are electronic statements available? | 200 / answered | Monthly. | Synthetic Statement Policy / 1 |
| statements-archive | How long do archived statements remain available? | 200 / answered | Twelve months. | Synthetic Statement Policy / 2 |
| statements-archive-paraphrase | For how many months can I access archived statements? | 200 / answered | Twelve months. | Synthetic Statement Policy / 2 |
| statements-older | How do I obtain statements older than the archive period? | 200 / answered | Submit a service request through the secure banking help centre. | Synthetic Statement Policy / 2 |
| complaints-ack | When is a complaint acknowledgement provided? | 200 / answered | Within two working days. | Synthetic Complaint Policy / 1 |
| complaints-ack-paraphrase | How soon will the bank acknowledge a complaint? | 200 / answered | Within two working days. | Synthetic Complaint Policy / 1 |
| complaints-escalation | When should an unresolved complaint be escalated? | 200 / answered | After seven working days to the designated grievance officer. | Synthetic Complaint Policy / 1 |
| accessibility-request | How can I request large print statements? | 200 / answered | Through the secure banking help centre. | Synthetic Accessibility Policy / 1 |
| accessibility-request-paraphrase | Where can I ask for a statement with bigger text? | 200 / answered | Request large print through the secure banking help centre. | Synthetic Accessibility Policy / 1 |
| accessibility-font | What is the minimum font size for large print statements? | 200 / answered | Eighteen points. | Synthetic Accessibility Policy / 1 |
| riverside-saturday | Is Riverside open on Saturdays? | 200 / answered | Open Saturdays from 10 AM to 2 PM. | Synthetic Riverside Branch FAQ / 1 |
| riverside-saturday-paraphrase | What are the Saturday opening times at Riverside? | 200 / answered | Open Saturdays from 10 AM to 2 PM. | Synthetic Riverside Branch FAQ / 1 |
| riverside-sunday | Does Riverside open on Sundays? | 200 / answered | Closed on Sundays. | Synthetic Riverside Branch FAQ / 1 |
| complaints-retention | How many years are complaint records retained? | 200 / not_found | Complaint acknowledgement/escalation do not establish record retention. | — |
| statements-csv | What CSV delimiter is used for electronic statement exports? | 200 / not_found | Policy documents PDF statements, not CSV export configuration. | — |
| unsupported-loan | What is the home loan interest rate? | 200 / not_found | No approved loan rates. | — |
| ambiguous-fee | What is the fee? | 200 / not_found | No product/service specified; multiple fee-free services exist. | — |
| ambiguous-time | How long does it take? | 200 / not_found | No service or event specified. | — |
| screen-otp | My OTP is 123456 | 422 / invalid_request | Reject before inference | — |
| screen-email | Please reply to person@example.test | 422 / invalid_request | Reject before inference | — |
| screen-empty |     | 422 / invalid_request | Reject before inference | — |
| failure-chroma | Is Riverside open on Saturdays? | 503 / unavailable | Safe retry response | — |
| failure-ollama | Is Riverside open on Saturdays? | 503 / unavailable | Safe retry response | — |
| updated-archive | How long do archived statements remain available? | 200 / answered | Twenty-four months in isolated updated document. | Synthetic Statement Policy / 2 |

For answered cases, the checker verifies the expected title/page against the actual stored chunk, checks the exact quote, and requires the reviewed fact fragments in both the answer and cited text. A test confirms that a real citation with an irrelevant quote fails. Offline model/retrieval doubles test the contract; the live run establishes behavior with actual Chroma/Ollama. This finite suite is not a general semantic-entailment proof.

Lifecycle checks: hypothetical archived-statement retention changes from twelve to twenty-four months, old chunk IDs disappear, the HTTP answer cites the revised page 2, and cleanup removes the temporary collection. A separate fingerprint comparison verifies the demo documents, IDs, metadata and vectors were not modified.
