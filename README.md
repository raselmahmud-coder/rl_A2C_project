# **Reinforcement Learning for Emerging Trend Detection in Tweets**

This project applies the **Advantage Actor-Critic (A2C)** reinforcement learning algorithm to detect emerging trends in tweets. The RL agent learns to optimize actions (post, edit, delete) based on engagement metrics such as likes, retweets, and quotes.



## **Project Structure**
```
A2C/
├── models/                    # Folder for saved models
├── data_loader.py             # Preprocessing and data loading
├── twitter_posting_env.py     # Custom RL environment
├── train.py                   # Training script for the RL agent
├── evaluate.py                # Evaluation script for the RL agent
└── data/
    ├── split_processed_tweets.csv   # Training dataset
    └── test_processed_tweets.csv    # Testing dataset
```



## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/raselmahmud-coder/rl_A2C_project.git
cd rl_A2C_project
```

### **2. Create a Conda Environment**
```bash
conda create -n rl_A2C_project python=3.8 -y
conda activate rl_A2C_project
```

### **3. Install Required Dependencies**
Install the Python libraries specified for the project:
```bash
pip install -r requirements.txt
```



## **How to Run the Project**

### **1. Train the RL Agent**
The training script trains the RL agent on the `split_processed_tweets.csv` dataset.
```bash
python train.py
```
- **Output**:
  - Trained model is saved as `a2c_tweet_trend_model` in the project directory.
  - `vec_normalize_stats.pkl` is saved to normalize observations and rewards for evaluation.


### **2. Evaluate the RL Agent**
After training, evaluate the RL agent on the `test_processed_tweets.csv` dataset to analyze its performance:
```bash
python evaluate.py
```
- **Outputs**:
  - Plots and metrics are saved in the `test_evaluation_results/` folder:
    - `total_and_average_reward.png`
    - `learning_curve.png`
    - `comparison_with_random_agent.png`
    - `convergence_check.png`
    - `smoothed_learning_curve.png`
  - Cumulative reward, average reward, and discounted reward are printed to the console.



## **Code Explanation**

### **1. Data Preprocessing (`data_loader.py`)**
- **Input**: Raw tweet data (`split_processed_tweets.csv`, `test_processed_tweets.csv`).
- **Preprocessing Steps**:
  - TF-IDF vectorization for tweet text.
  - Normalization of engagement metrics (likes, retweets, quotes).
- **Output**: Features and rewards for the RL environment.



### **2. RL Environment (`twitter_posting_env.py`)**
The `TwitterPostingEnv` simulates an environment for the RL agent:
- **State Space**:
  - Combined features from TF-IDF and engagement metrics.
- **Action Space**:
  - `0`: Post tweet.
  - `1`: Edit tweet.
  - `2`: Delete tweet.
- **Reward Function**:
  - Engagement-based rewards:
    - Post: \( R = \text{engagement} \times 1.05 \)
    - Edit: \( R = \text{engagement} \times 1.02 \)
    - Delete: \( R = \text{engagement} \times 0.5 \)



### **3. Training (`train.py`)**
- **Agent**: A2C (Advantage Actor-Critic).
- **Features**:
  - `VecNormalize`: Normalizes observations and rewards.
  - Hyperparameters:
    - `learning_rate=0.0001`
    - `ent_coef=0.05`
    - `total_timesteps=200,000`
- **Output**:
  - Trained model (`a2c_tweet_trend_model`).
  - Normalization statistics (`vec_normalize_stats.pkl`).



### **4. Evaluation (`evaluate.py`)**
- Loads the trained agent and evaluates it on the test dataset.
- **Metrics**:
  - Total reward, average reward, discounted reward.
  - Comparison with a random agent.
  - Convergence check using reward variance.
- **Output**:
  - Saved plots in `test_evaluation_results/`.
  - Console log of performance metrics.



## **Results**
- The RL agent outperformed the random agent consistently.
- Training phases showed incremental improvements in cumulative and average rewards.
- Evaluation metrics demonstrated convergence and stability in the agent’s performance.



## **Future Improvements**
1. Incorporate additional features like sentiment analysis or hashtag relevance.
2. Compare A2C with other RL algorithms (e.g., PPO, DQN).
3. Validate the model on larger or more diverse datasets.



## **Contact**
For any questions or issues, feel free to contact:
- **Name**: Md Rasel Mahmud
- **Email**: raselmahmud@mail.ustc.edu.cn
