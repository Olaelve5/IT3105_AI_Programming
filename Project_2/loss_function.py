from MuZeroNet import MuZeroNet
import jax.numpy as jnp
import jax


def normalize_value(x):
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + 0.001 * x


def loss_function(params, model: MuZeroNet, batch):
    obs = batch["observations"]

    # Initial Step
    hidden_state, raw_policy_scores, predicted_value = model.apply(
        params, obs, method=model.initial_inference
    )

    target_policy = batch["target_policies"][:, 0]
    target_value = batch["target_values"][:, 0]

    action_probs = jax.nn.log_softmax(raw_policy_scores, axis=-1)

    # Calculate initial losses using SUM (not mean)
    policy_loss = -jnp.sum(target_policy * action_probs, axis=-1)
    value_loss = (
        normalize_value(predicted_value.squeeze(-1)) - normalize_value(target_value)
    ) ** 2

    # Initialize accumulators using SUMS
    total_loss = jnp.sum(policy_loss + value_loss)
    total_policy_loss = jnp.sum(policy_loss)
    total_value_loss = jnp.sum(value_loss)
    total_reward_loss = 0.0
    total_discount_loss = 0.0

    unroll_steps = batch["target_policies"].shape[1]

    # Recurrent Steps
    for i in range(1, unroll_steps):
        action = batch["actions"][:, i - 1]
        hidden_state = hidden_state * 0.5 + jax.lax.stop_gradient(hidden_state) * 0.5

        hidden_state, pred_reward, pred_discount, raw_policy_scores, pred_value = (
            model.apply(params, hidden_state, action, method=model.recurrent_inference)
        )

        target_reward = batch["target_rewards"][:, i - 1]
        target_discount = batch["target_discounts"][:, i - 1]
        target_policy = batch["target_policies"][:, i]
        target_value = batch["target_values"][:, i]

        # Calculate step losses using SUM (not mean)
        reward_loss = (pred_reward.squeeze(-1) - target_reward) ** 2
        discount_loss = (pred_discount.squeeze(-1) - target_discount) ** 2

        action_probs = jax.nn.log_softmax(raw_policy_scores, axis=-1)
        step_policy_loss = -jnp.sum(target_policy * action_probs, axis=-1)
        step_value_loss = (
            normalize_value(pred_value.squeeze(-1)) - normalize_value(target_value)
        ) ** 2

        step_total_loss = (
            reward_loss + discount_loss + step_policy_loss + step_value_loss
        )
        masked_loss = step_total_loss * target_discount

        # Accumulate sums
        total_loss += jnp.sum(masked_loss)
        total_reward_loss += jnp.sum(reward_loss * target_discount)
        total_discount_loss += jnp.sum(discount_loss * target_discount)
        total_policy_loss += jnp.sum(step_policy_loss * target_discount)
        total_value_loss += jnp.sum(step_value_loss * target_discount)

    scale = batch["observations"].shape[0] * (unroll_steps + 1)

    # Create the dictionary of auxiliary metrics
    metrics = {
        "total": total_loss / scale,
        "policy": total_policy_loss / scale,
        "value": total_value_loss / scale,
        "reward": total_reward_loss / scale,
        "discount": total_discount_loss / scale,
    }

    # metrics is for logging
    return total_loss / scale, metrics
