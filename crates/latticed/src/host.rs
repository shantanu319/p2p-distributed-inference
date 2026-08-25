//! What this machine advertises about itself.
//!
//! Memory here is the physical total, which is advisory: §5's planner uses a
//! measured, ceiling-adjusted figure, not this one.

pub struct HostFacts {
    pub name: String,
    pub platform: String,
    pub total_memory: u64,
}

pub fn detect() -> HostFacts {
    HostFacts {
        name: hostname(),
        platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        total_memory: total_memory(),
    }
}

fn hostname() -> String {
    let mut buf = [0i8; 256];
    // SAFETY: buf is a valid writable buffer of the length passed.
    let rc = unsafe { libc::gethostname(buf.as_mut_ptr(), buf.len()) };
    if rc != 0 {
        return "unknown".into();
    }
    // SAFETY: gethostname NUL-terminates within the buffer on success.
    let name = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) };
    let name = name.to_string_lossy();
    // Trim the mDNS suffix so "studio.local" and "studio" read the same.
    name.strip_suffix(".local")
        .unwrap_or(&name)
        .to_owned()
}

#[cfg(target_os = "macos")]
fn total_memory() -> u64 {
    let mut value = 0u64;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: hw.memsize is a u64 sysctl; value and len match its size.
    let rc = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&raw mut value).cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 { value } else { 0 }
}

#[cfg(not(target_os = "macos"))]
fn total_memory() -> u64 {
    let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") else {
        return 0;
    };
    meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|kb| kb.parse::<u64>().ok())
        .map_or(0, |kb| kb * 1024)
}

#[cfg(test)]
mod tests {
    #[test]
    fn this_machine_reports_plausible_facts() {
        let facts = super::detect();
        assert!(!facts.name.is_empty());
        assert!(!facts.name.ends_with(".local"));
        assert!(facts.platform.contains('-'));
        assert!(
            facts.total_memory >= 1 << 30,
            "implausible total memory: {}",
            facts.total_memory
        );
    }
}
