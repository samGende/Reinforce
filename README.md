# Reimplementation of Reinforce++ To finetune llama-1b for GSM8K.

https://arxiv.org/html/2501.03262v1

## Roadmap
- [ x ] KL Penatly
- [ ] SFT for GSM8K
- [ ] Toggle sparse rewards $r(s_t,a_t) = I(s_T=[EOS])r(x,y) - bKL(t)$
