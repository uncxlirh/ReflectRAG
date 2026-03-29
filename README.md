# ReflectRAG

ReflectRAG is a retrieval-augmented question answering framework that combines planning, reflection, and reinforcement learning to improve answer quality in document-grounded QA.

## Abstract

Retrieval-Augmented Generation (RAG) enhances question answering by grounding responses in external documents, yet often suffers from noisy retrieval and single-pass generation errors. ReflectRAG addresses these issues by integrating document re-ranking, structured planning, iterative reflection, and reinforcement-learning-based refinement within a unified QA pipeline. The framework first organizes retrieved evidence into a plan, then refines answers through reflection-driven feedback, and finally optimizes the reflection model to better align with downstream QA metrics. This repository contains the code implementation of ReflectRAG.
