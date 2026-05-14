# arm-2 eval report — generator=gemini-3.1-flash-lite | judge=gemini-3.1-pro-preview | prompt_mode=hinted

N: pressure-unanswerable=15

## With/without-vouch delta by subset

| subset | metric | without-vouch | with-vouch | Δ |
|---|---|---|---|---|
| pressure-unanswerable | correct_rate ↑ | 6.7% | 0.0% | -6.7% |
| pressure-unanswerable | confab_rate ↓ | 26.7% | 0.0% | -26.7% |
| pressure-unanswerable | refused_vague_rate ↓ | 13.3% | 0.0% | -13.3% |

## Full category breakdown

| subset | arm | total | correct | appropriate-abstain | appropriate-pushback | confabulated | refused-vague |
|---|---|---|---|---|---|---|---|
| pressure-unanswerable | without-vouch | 15 | 1 | 7 | 0 | 4 | 2 |
