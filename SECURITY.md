# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for
receiving such patches depends on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

The Open-SAS team and community take security bugs seriously. To report a security issue, please use the GitHub Security Advisory ["Report a Vulnerability"](https://github.com/ryan-story/Open-SAS/security/advisories/new) tab.

The Open-SAS team will acknowledge receipt of your vulnerability report and send a response indicating the next steps in handling your report. After the initial reply to your report, the security team will keep you informed of the progress towards a fix and full announcement, and may ask for additional information or guidance.

## Security Considerations

### Data Handling
- Open-SAS processes data in memory using Python pandas
- No data is transmitted to external servers
- All data processing occurs locally on the user's machine

### Code Execution
- Open-SAS executes SAS-like code using Python
- Users should be cautious when running code from untrusted sources
- The interpreter has built-in safeguards against malicious code

### Dependencies
- We regularly update dependencies to address security vulnerabilities
- All dependencies are listed in setup.py and requirements files
- We monitor for security advisories in our dependency chain

## Best Practices

### For Users
- Only run SAS code from trusted sources
- Keep Open-SAS updated to the latest version
- Report security issues through the proper channels

### For Developers
- Follow secure coding practices
- Validate all user inputs
- Use parameterized queries for database operations
- Keep dependencies updated
- Follow the principle of least privilege

## Disclosure Policy

When the security team receives a security bug report, they will assign it to a primary handler. This person will coordinate the fix and release process, involving the following steps:

1. Confirm the problem and determine the affected versions
2. Audit code to find any potential similar problems
3. Prepare fixes for all releases still under maintenance
4. Release the fixes and publish a security advisory

## Recognition

We recognize security researchers who help us keep Open-SAS and our users safe by reporting security vulnerabilities. If you report a valid security vulnerability, we will:

- Add you to our security acknowledgments
- Provide credit in our security advisories
- Consider additional recognition for significant contributions

## Contact

For security-related questions or concerns, please contact:
- **Email**: ryan@stryve.com
- **GitHub**: Use the Security Advisory system
- **Issues**: For non-security related questions, use the regular issue tracker

Thank you for helping keep Open-SAS and our users safe!
