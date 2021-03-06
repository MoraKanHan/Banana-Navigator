# Project 1: Navigation
<h3>Introduction</h3>
For this project, a DQN and Double DQN agents were implemented and trained to navigate (and collect bananas!) in a large, square world. This project is part of the Deep Reinforcement Learning Nanodegree program.

<img class="image--26lOQ" src="https://video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif" alt="" width="400px">

<h3>Rewards</h3>
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of DQN and Double DQN agents is to collect as many yellow bananas as possible while avoiding blue bananas.

<h3>Environment</h3>
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
<ul>
<li> 0 - move forward.
<li> 1 - move backward.
<li> 2 - turn left.
<li> 3 - turn right.
</ul>
The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

# Getting Started
<h3>Download DRLND repository</h3>
<p>To set up your python environment to run the code in this repository, follow the instructions below.</p>
<p>Create (and activate) a new environment with Python 3.6.</p>
<ol>
<li>
<p>Create (and activate) a new environment with Python 3.6.</p>
<ul>
<li><strong>Linux</strong> or <strong>Mac</strong>:</li>
</ul>
<div class="highlight highlight-source-shell position-relative"><pre>conda create --name drlnd python=3.6
<span class="pl-c1">source</span> activate drlnd</pre><div class="zeroclipboard-container position-absolute right-0 top-0">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn js-clipboard-copy m-2 p-0 tooltipped-no-delay" data-copy-feedback="Copied!" data-tooltip-direction="w" value="conda create --name drlnd python=3.6
source activate drlnd
" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-paste js-clipboard-clippy-icon m-2">
    <path fill-rule="evenodd" d="M5.75 1a.75.75 0 00-.75.75v3c0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75v-3a.75.75 0 00-.75-.75h-4.5zm.75 3V2.5h3V4h-3zm-2.874-.467a.75.75 0 00-.752-1.298A1.75 1.75 0 002 3.75v9.5c0 .966.784 1.75 1.75 1.75h8.5A1.75 1.75 0 0014 13.25v-9.5a1.75 1.75 0 00-.874-1.515.75.75 0 10-.752 1.298.25.25 0 01.126.217v9.5a.25.25 0 01-.25.25h-8.5a.25.25 0 01-.25-.25v-9.5a.25.25 0 01.126-.217z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-text-success d-none m-2">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<ul>
<li><strong>Windows</strong>:</li>
</ul>
<div class="highlight highlight-source-shell position-relative"><pre>conda create --name drlnd python=3.6 
activate drlnd</pre><div class="zeroclipboard-container position-absolute right-0 top-0">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn js-clipboard-copy m-2 p-0 tooltipped-no-delay" data-copy-feedback="Copied!" data-tooltip-direction="w" value="conda create --name drlnd python=3.6 
activate drlnd
" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-paste js-clipboard-clippy-icon m-2">
    <path fill-rule="evenodd" d="M5.75 1a.75.75 0 00-.75.75v3c0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75v-3a.75.75 0 00-.75-.75h-4.5zm.75 3V2.5h3V4h-3zm-2.874-.467a.75.75 0 00-.752-1.298A1.75 1.75 0 002 3.75v9.5c0 .966.784 1.75 1.75 1.75h8.5A1.75 1.75 0 0014 13.25v-9.5a1.75 1.75 0 00-.874-1.515.75.75 0 10-.752 1.298.25.25 0 01.126.217v9.5a.25.25 0 01-.25.25h-8.5a.25.25 0 01-.25-.25v-9.5a.25.25 0 01.126-.217z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-text-success d-none m-2">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
    </clipboard-copy>
  </div></div>
</li>

<li>
<p>Clone the repository (if you haven't already!), and navigate to the <code>python/</code> folder.  Then, install several dependencies.</p>
</li>
</ol>
<div class="highlight highlight-source-shell position-relative"><pre>git clone https://github.com/udacity/deep-reinforcement-learning.git
<span class="pl-c1">cd</span> deep-reinforcement-learning/python
pip install <span class="pl-c1">.</span></pre><div class="zeroclipboard-container position-absolute right-0 top-0">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn js-clipboard-copy m-2 p-0 tooltipped-no-delay" data-copy-feedback="Copied!" data-tooltip-direction="w" value="git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-paste js-clipboard-clippy-icon m-2">
    <path fill-rule="evenodd" d="M5.75 1a.75.75 0 00-.75.75v3c0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75v-3a.75.75 0 00-.75-.75h-4.5zm.75 3V2.5h3V4h-3zm-2.874-.467a.75.75 0 00-.752-1.298A1.75 1.75 0 002 3.75v9.5c0 .966.784 1.75 1.75 1.75h8.5A1.75 1.75 0 0014 13.25v-9.5a1.75 1.75 0 00-.874-1.515.75.75 0 10-.752 1.298.25.25 0 01.126.217v9.5a.25.25 0 01-.25.25h-8.5a.25.25 0 01-.25-.25v-9.5a.25.25 0 01.126-.217z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-text-success d-none m-2">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
    </clipboard-copy>
  </div></div>
  <ol start="4">
<li>Create an <a href="http://ipython.readthedocs.io/en/stable/install/kernel_install.html" rel="nofollow">IPython kernel</a> for the <code>drlnd</code> environment.</li>
</ol>
<div class="highlight highlight-source-shell position-relative"><pre>python -m ipykernel install --user --name drlnd --display-name <span class="pl-s"><span class="pl-pds">"</span>drlnd<span class="pl-pds">"</span></span></pre><div class="zeroclipboard-container position-absolute right-0 top-0">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn js-clipboard-copy m-2 p-0 tooltipped-no-delay" data-copy-feedback="Copied!" data-tooltip-direction="w" value="python -m ipykernel install --user --name drlnd --display-name &quot;drlnd&quot;
" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-paste js-clipboard-clippy-icon m-2">
    <path fill-rule="evenodd" d="M5.75 1a.75.75 0 00-.75.75v3c0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75v-3a.75.75 0 00-.75-.75h-4.5zm.75 3V2.5h3V4h-3zm-2.874-.467a.75.75 0 00-.752-1.298A1.75 1.75 0 002 3.75v9.5c0 .966.784 1.75 1.75 1.75h8.5A1.75 1.75 0 0014 13.25v-9.5a1.75 1.75 0 00-.874-1.515.75.75 0 10-.752 1.298.25.25 0 01.126.217v9.5a.25.25 0 01-.25.25h-8.5a.25.25 0 01-.25-.25v-9.5a.25.25 0 01.126-.217z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-text-success d-none m-2">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
    </clipboard-copy>
  </div></div>
  <ol start="5">
<li>Before running code in a notebook, change the kernel to match the <code>drlnd</code> environment by using the drop-down <code>Kernel</code> menu.</li>
</ol>
<p><a target="_blank" rel="noopener noreferrer" href="https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png"><img src="https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png" alt="Kernel" title="Kernel" style="max-width:100%;"></a></p>


<h3>Download Unity Environment</h3>
Download the environment from one of the links below. You need only select the environment that matches your operating system:

<ul>
<li>Linux: <a target="_blank" href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip">click here</a></li>
<li>Mac OSX: <a target="_blank" href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip">click here</a></li>
<li>Windows (32-bit): <a target="_blank" href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip">click here</a></li>
<li>Windows (64-bit): <a target="_blank" href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip">click here</a></li>
</ul>
<p>(<em>For Windows users</em>) Check out <a target="_blank" href="https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64">this link</a> if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.</p>

<p>(<em>For AWS</em>) If you'd like to train the agent on AWS (and have not <a target="_blank" href="https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md">enabled a virtual screen</a>), then please use <a target="_blank" href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip">this link</a> to obtain the "headless" version of the environment.  You will <strong>not</strong> be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (<em>To watch the agent, you should follow the instructions to <a target="_blank" href="https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md">enable a virtual screen</a>, and then download the environment for the <strong>Linux</strong> operating system above.</em>)</p>

Place the file in the DRLND GitHub repository, in the <code>p1_navigation/</code> folder, and unzip (or decompress) the file.

# Instructions
Follow the instructions in <code>DQN/</code> and <code>DoubleDQN/</code> folders to get started with training the agents agent!

# Implementation Details
All implementation details and results are found in <code>Report/</code> folder

# References
<ul>
    <li>[1]: V. Mnih et al., "Human-level control through deep reinforcement learning", Nature, vol. 518, no. 7540, pp. 529-533, 2015. Available: 10.1038/nature14236 [Accessed 3 September 2021].
<li>[2]: U. Technologies, "Machine Learning Agents | Unity", Unity, 2021. [Online]. Available: https://unity.com/products/machine-learning-agents. [Accessed: 03- Sep- 2021]. 
<li>[3]: R. Sutton and A. Barto, Reinforcement Learning, 2nd ed. 2019.
<li>[4]: H. van Hasselt, A. Guez and D. Silver, "Deep Reinforcement Learning with Double Q-learning", 2015. [Accessed 3 September 2021].
<li>[5]: M. Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning", 2021. [Accessed 3 September 2021]. 
</ul>

