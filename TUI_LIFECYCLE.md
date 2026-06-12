# Terminal user interface (TUI) app `lqh` lifecycles

We want the TUI app to include the following two features:

## Background task with actual monitoring
Currently when we start a long running job (training of eval in the cloud or remote) I see that 'background' tasks are being "monitored". However, it does not seem to work. I manually have to ask the agent to run a training_status to see if the training has completed. Please fix this.
All background tasks, should be regularly polled with a conservative timeout (e.g. 60 seconds) to check if they have completed. If they have completed, we should invoke the agent and update the TUI with the new information. This way, the user does not have to manually ask for updates, and the TUI will always reflect the current state of background tasks.

## "Close laptop" tolerance
The TUI app should tolerate any disconnect via wifi disconnect or laptop sleep.
This spans from the orchestration model to data generation and background task monitoring. If the user disconnects, the TUI should not crash or lose state. When the user reconnects or the user opens the laptop again, the TUI should automatically reconnect to the backend and resume monitoring any background tasks or ongoing processes. This way, users can safely close their laptops or disconnect from wifi without worrying about losing progress or having to restart the TUI app.
These disconnects or sleep time probably materialize as LLM API timeouts or connection errors. The TUI app should catch these exceptions and handle them gracefully, allowing the user to reconnect without losing state or progress.

Note: we should implement this without digging into operating system level events but make the app tolerant.

- Data gen should continue as usual
- Job monitoring should continue as usual, and the TUI should update when the user reconnects or opens the laptop again.
- normal chat interactions with the orchestrator should be tolerant to disconnects and reconnects, allowing the user to continue their work without interruption.

Note: The feature on top, properly implementing the background task monitoring, needs to go hand-in-hand with this feature.

Note: We generally probably want to also implement some background time tracker to detect these events by looking at the system time.

Feature: we probably want to retry 3 times with a backoff (3s, 20s, 60s) before giving up and showing an error to the user. This way, we can handle temporary disconnects without immediately showing an error, but we also don't want to keep retrying indefinitely if there is a persistent issue.
We should also include a "/reconnect" command that does not do anything in normal operations but starts a new reconnect attempt with 3 backoffs when the user issues the command. This way, if the user knows they have a bad connection, they can proactively trigger a reconnect attempt without having to wait for the automatic retries to kick in.
The error message should include the info that the user can run "/reconnect" to attempt to reconnect.
