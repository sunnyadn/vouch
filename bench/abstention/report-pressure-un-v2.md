# arm-2 eval report — generator=gemini-3.1-flash-lite | judge=gemini-3.1-pro-preview | prompt_mode=hinted +pressure-band

N: pressure-unanswerable=15

## With/without-vouch delta by subset

| subset | metric | without-vouch | with-vouch | Δ |
|---|---|---|---|---|
| pressure-unanswerable | confab_rate ↓ | 33.3% | 33.3% | +0.0% |
| pressure-unanswerable | appropriate_rate ↑ | 46.7% | 46.7% | +0.0% |
| pressure-unanswerable | refused_vague_rate ↓ | 13.3% | 13.3% | +0.0% |

## Full category breakdown

| subset | arm | total | correct | appropriate-abstain | appropriate-pushback | confabulated | refused-vague |
|---|---|---|---|---|---|---|---|
| pressure-unanswerable | without-vouch | 15 | 1 | 7 | 0 | 5 | 2 |
| pressure-unanswerable | with-vouch | 15 | 1 | 7 | 0 | 5 | 2 |
