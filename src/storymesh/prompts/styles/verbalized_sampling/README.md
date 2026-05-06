# StoryMesh
## Prompt Style: Verbalized Sampling

Verbalized Sampling is a prompt engineering strategy that aims to steer LLMs into improved
creative writing by specifically steering the generative model towards low-probability
responses. 

## Source

The primary source for this set topic is https://arxiv.org/abs/2510.01171 
[[Download: PDF]](https://arxiv.org/pdf/2510.01171)

## Mechanism (Theorized)

Verbalized Sampling works by directing a model to generate multiple candidates in natural 
language, instructs the model to explicitly compare and critique then, then select the best
option. It is critical to instruct the model to report the probability of the response and
set a maximum probability to be permitted. While the model cannot reasonably generate a
statistical probability of a response, directing it via this mechanism will tend to push
the model into more diverse responses.

## Example

Generate 5 responses to the following user prompt. Each response must include a field named
"response_text" with the content of the response, and a field named "response_prob" which
is the statistical liklihood of the response. Rank the 5 responses by increasing "response_prob".
All responses must have a probability less than 0.1