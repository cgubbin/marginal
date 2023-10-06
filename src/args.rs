use std::env;
use std::ffi::OsString;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process;
use std::sync::Arc;

use crate::app;
use crate::config;
use crate::Result;

pub struct Args(Arc<ArgsInner>);

pub struct ArgsInner {
    /// The command we want to execute
    command: Command,
    /// The number of threads to use
    threads: usize,
    /// Working directories provided at the command line
    paths: Vec<PathBuf>,
}

pub enum Command {
    Calibrate,
}

impl Args {
    pub fn parse() -> Result<Self> {
        // we collect arguments parsed on the command line
        let early_matches = ArgMatches::new(clap_matches(env::args_os())?);

        // TODO impl logging etc and initialize

        let matches = early_matches.reconfigure()?;

        matches.to_args()
    }

    pub fn command(&self) -> &Command {
        &self.0.command
    }

    /// Return the paths found in the command line arguments. This is
    /// guaranteed to be non-empty. In the case where no explicit arguments are
    /// provided, a single default path is provided automatically.
    fn paths(&self) -> &[PathBuf] {
        &self.0.paths
    }
}

/// `ArgMatches` wraps `clap::ArgMatches` and provides semantic meaning to
/// the parsed arguments.
#[derive(Clone, Debug)]
struct ArgMatches(clap::ArgMatches);

impl ArgMatches {
    fn new(arg_matches: clap::ArgMatches) -> Self {
        Self(arg_matches)
    }
}

impl ArgMatches {
    fn contains_id(&self, id: &str) -> bool {
        self.0.contains_id(id)
    }

    fn reconfigure(self) -> Result<Self> {
        if self.contains_id("no-config") {
            return Ok(self);
        }

        let mut args = config::args();
        if args.is_empty() {
            return Ok(self);
        }

        let mut cliargs = env::args_os();
        if let Some(bin) = cliargs.next() {
            args.insert(0, bin);
        }
        args.extend(cliargs);
        Ok(ArgMatches(clap_matches(args)?))
    }

    fn to_args(self) -> Result<Args> {
        todo!()
    }
}

fn clap_matches<I, T>(args: I) -> Result<clap::ArgMatches>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let err = match app::app().try_get_matches_from(args) {
        Ok(matches) => return Ok(matches),
        Err(err) => err,
    };
    if err.use_stderr() {
        return Err(err.into());
    }
    // Explicitly ignore any error returned by write!. The most likely error
    // at this point is a broken pipe error, in which case, we want to ignore
    // it and exit quietly.
    //
    // (This is the point of this helper function. clap's functionality for
    // doing this will panic on a broken pipe error.)
    let _ = write!(std::io::stdout(), "{}", err);
    process::exit(0);
}
