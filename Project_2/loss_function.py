import MuZeroNet
import jax.numpy as jnp
import jax


def loss_function(params, model: MuZeroNet, batch):
    obs = batch["observations"]

    # First get the loss for the first step
    hidden_state, raw_policy_scores, predicted_value = model.apply(
        params, obs, method=model.initial_inference
    )

    target_policy = batch["target_policies"][:, 0]
    target_value = batch["target_values"][:, 0]

    action_probs = jax.nn.log_softmax(raw_policy_scores, axis=-1)
    policy_loss = -jnp.sum(target_policy * action_probs, axis=-1).mean()
    value_loss = jnp.mean((predicted_value.squeeze(-1) - target_value) ** 2)

    total_loss = policy_loss + value_loss
    unroll_steps = batch["target_policies"].shape[1]

    # Then get the loss for the rest of the steps
    for i in range(1, unroll_steps):
        action = batch["actions"][:, i - 1]

        hidden_state = jax.lax.stop_gradient(hidden_state) 
        hidden_state, pred_reward, raw_policy_scores, pred_value = model.apply(
            params, hidden_state, action, method=model.recurrent_inference
        )

        target_reward = batch["target_rewards"][:, i - 1]
        target_policy = batch["target_policies"][:, i]
        target_value = batch["target_values"][:, i]

        reward_loss = jnp.mean((pred_reward.squeeze(-1) - target_reward) ** 2)
        action_probs = jax.nn.log_softmax(raw_policy_scores, axis=-1)
        policy_loss = -jnp.sum(target_policy * action_probs, axis=-1).mean()
        value_loss = jnp.mean((pred_value.squeeze(-1) - target_value) ** 2)

        total_loss = total_loss + reward_loss + policy_loss + value_loss

    # Scale/normalize the loss
    return total_loss / (unroll_steps - 1)
