# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| `main` branch (latest) | ✅ |
| Older commits / tags | ❌ — please upgrade |

RoLLM is a pure Luau library with no server-side components, network listeners, or persistent storage. The attack surface is limited to the code you run inside your Roblox experience.

---

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please report privately using one of the following methods:

1. **GitHub Security Advisory** — go to the [Security tab](https://github.com/rustyspottedcatt/RoLLM/security/advisories/new) and open a private advisory.
2. **Direct contact** — reach the maintainer via the contact information on their [GitHub profile](https://github.com/rustyspottedcatt).

---

## What to Include

- Affected version or commit SHA
- Minimal reproduction (Luau snippet or place file)
- Expected behaviour vs. actual behaviour
- Estimated impact (e.g., data exfiltration, remote code execution, denial of service)
- Any Roblox place, model, or dependency context needed to reproduce

---

## Response Timeline

| Stage | Target |
|---|---|
| Acknowledgement | Within 72 hours |
| Initial assessment | Within 7 days |
| Fix or mitigation | Best-effort; critical issues prioritised |
| Public disclosure | Coordinated with reporter after fix is available |

---

## Scope

Issues likely **in scope**:

- Arbitrary code execution via malicious corpus or vocab input
- Denial of service caused by unbounded computation or memory growth from crafted input
- Unsafe `loadstring` / `HttpService` usage in BPE vocab loading

Issues likely **out of scope**:

- Model output quality or hallucinations (not a security issue)
- Roblox platform bugs unrelated to RoLLM code
- Vulnerabilities in transitive Wally dependencies (report upstream)
