#!/usr/bin/env python3
import subprocess, sys, json, os

PY = sys.executable or 'python3'
CLI = os.path.join(os.path.dirname(__file__), '..', 'rental_threshold_calculator_dynamic.py')

def run(cmd):
    print('='.ljust(80, '='))
    print('CMD:', ' '.join(cmd))
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        print('Exit:', p.returncode)
        if p.stdout:
            print('STDOUT:\n', p.stdout[:2000])
        if p.stderr:
            print('STDERR:\n', p.stderr[:2000])
        return p.returncode
    except Exception as e:
        print('ERROR:', e)
        return 1

def main():
    rc = 0
    # Scenario 1: Small DP, compound Poisson (lam=1.5), simple prices
    rc |= run([PY, CLI,
               '--inventory','3','--periods','5','--cost','50',
               '--arrival-rate','1.5','--prices','45,50,55,60'])

    # Scenario 2: Static only sanity
    rc |= run([PY, CLI,
               '--inventory','10','--periods','30','--cost','70',
               '--arrival-rate','0.8','--prices','50,55,57,58,60,65,68,70',
               '--static-only'])

    # Scenario 3: Live decision, per-offer with mid horizon
    rc |= run([PY, CLI,
               '--inventory','5','--periods','10','--cost','60',
               '--arrival-rate','2.0','--prices','55,58,60,62,65,70',
               '--live','--offer-price','62','--current-period','4','--current-inventory','4'])

    # Scenario 4: High penalty pushing thresholds up
    rc |= run([PY, CLI,
               '--inventory','8','--periods','10','--cost','60',
               '--arrival-rate','0.5','--prices','50,55,58,60,62,65,70',
               '--penalty-alpha','50','--penalty-beta','500'])

    # Scenario 5: Large lambda to exercise K>1 paths
    rc |= run([PY, CLI,
               '--inventory','4','--periods','6','--cost','55',
               '--arrival-rate','3.0','--prices','50,52,54,56,58,60,65'])

    print('='*80)
    print('Overall exit code:', rc)
    sys.exit(rc)

if __name__ == '__main__':
    main()

