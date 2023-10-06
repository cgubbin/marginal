use clap::{Arg, Command};

pub(crate) fn app() -> Command {
    let mut command = Command::new("marginal")
        .about("calibrate and process real time sensor data")
        .subcommand_required(true);

    for arg in all_args_and_flags() {
        command = command.arg(arg);
    }

    command
}

// Build the arg-list
fn all_args_and_flags() -> Vec<Arg> {
    todo!()
}
