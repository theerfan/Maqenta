# Maqenta

## [Magenta](https://github.com/magenta/magenta), but in Quantum.

## Roadmap
1. Learn about GANs
   * [x] Watch [Ian Goodfellow: Generative Adversarial Networks](https://www.youtube.com/watch?v=HGYYEUSm-0Q). 
   * [x] Watch [Seth Lloyd: Quantum Generative Adversarial Networks](https://www.youtube.com/watch?v=5nfN8xT3Z8g).
   * [x] Go through [PennyLane's Tutorial of QuGANs](https://pennylane.ai/qml/demos/tutorial_QGAN.html).
   * [x] Chris Olah - [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
   * [ ] Go through [GANSynth tutorial on Magenta](https://magenta.tensorflow.org/gansynth).
   * [ ] Port GANSynth to Quantum.

2. Learn about RNNs & LSTMs 
   * [x] Implement [Quantum Long Short-Term memory](https://arxiv.org/abs/2009.01783) in PennyLane.
   * [x] Implement a music generation algorithm using the above.
   * [ ] Go through [RNN music generation tutorial on Magenta](https://magenta.tensorflow.org/2016/06/10/recurrent-neural-network-generation-tutorial).
   * [ ] Port the above to Quantum.

3. Learn more about other QML Methods that could be possible for generating music.
   * [x] [Quanvolutional Neural Networks](https://pennylane.ai/qml/demos/tutorial_quanvolution.html).
   * [x] [Quantum transfer learning](https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html).
   * [ ] [Quantum models as Fourier series](https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series.html).
   * [ ] [Learning to learn with quantum neural networks](https://pennylane.ai/qml/demos/learning2learn.html).
   * [ ] [Quantum support vector machine for big data classification](https://arxiv.org/abs/1307.0471).
 
4. Learn Music Theory to be able to find even more ideas for generating Quantum Music.
   * [x] [Learn music theory in half an hour](https://www.youtube.com/watch?v=rgaTLrZGlk0).
   * [ ] [Music And Measure Theory - 3Blue1Brown](https://www.youtube.com/watch?v=cyW5z-M2yzw).
    
 5. Read these papers
   * [x] [Recurrent Quantum Neural Networks](https://arxiv.org/abs/2006.14619).
   * [ ] [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661).
   * [ ] [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196).
   * [ ] [Adversarial Audio Synthesis](https://arxiv.org/abs/1802.04208).
   * [ ] [GANSynth: Adversarial Neural Audio Synthesis](https://arxiv.org/abs/1902.08710).

7. Current Development
   * [ ] Develop the code for EQGAN.
   * [ ] Change the measurements from global to local in the QuGAN module, because it gets trapped inside a barren plateu very quickly.
   * [ ] The Issue of monotony (same offset between notes). 
