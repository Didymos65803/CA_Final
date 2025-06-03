#!/usr/bin/env python3
"""
simple_test.py
=============
Simple test to verify the N-body kernels work correctly
"""

import numpy as np
import matplotlib.pyplot as plt
import time

def test_kernels():
    """Test all three N-body kernels"""
    print("Testing N-body kernels...")
    
    try:
        from force_kernel import direct_omp, bh_omp
        from fmm_kernel import fmm_omp
        print("✓ All modules imported successfully")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("Please run: python rebuild_and_test.py first")
        return False
    
    # Create test particles
    N = 100
    np.random.seed(42)  # For reproducible results
    
    # Random distribution matching fmm_scaling_test.py
    x = (np.random.random(N) - 0.5) * 100.0
    y = (np.random.random(N) - 0.5) * 100.0
    m = np.random.uniform(1.0, 5.0, N)
    
    print(f"\nTesting with {N} particles...")
    print(f"Position range: x=[{x.min():.1f}, {x.max():.1f}], y=[{y.min():.1f}, {y.max():.1f}]")
    print(f"Mass range: [{m.min():.1f}, {m.max():.1f}]")
    
    # Test parameters
    G = 1.0
    soft = 0.01
    domain = 100.0
    theta = 0.5
    
    results = {}
    
    # Test Direct method
    print("\n1. Testing Direct method...")
    try:
        t0 = time.time()
        ax_direct, ay_direct = direct_omp(x, y, m, G, soft)
        t_direct = time.time() - t0
        
        print(f"   ✓ Direct: {t_direct:.4f} s")
        print(f"   Force range: ax=[{ax_direct.min():.2e}, {ax_direct.max():.2e}]")
        print(f"                ay=[{ay_direct.min():.2e}, {ay_direct.max():.2e}]")
        results['direct'] = (ax_direct, ay_direct, t_direct)
    except Exception as e:
        print(f"   ✗ Direct failed: {e}")
        return False
    
    # Test Barnes-Hut method
    print("\n2. Testing Barnes-Hut method...")
    try:
        t0 = time.time()
        ax_bh, ay_bh = bh_omp(x, y, m, domain, theta, G, soft)
        t_bh = time.time() - t0
        
        print(f"   ✓ Barnes-Hut: {t_bh:.4f} s")
        print(f"   Force range: ax=[{ax_bh.min():.2e}, {ax_bh.max():.2e}]")
        print(f"                ay=[{ay_bh.min():.2e}, {ay_bh.max():.2e}]")
        
        # Calculate error relative to direct
        error_bh = np.mean(np.sqrt((ax_bh - ax_direct)**2 + (ay_bh - ay_direct)**2) / 
                          (np.sqrt(ax_direct**2 + ay_direct**2) + 1e-10))
        print(f"   Relative error vs Direct: {error_bh:.2e}")
        results['bh'] = (ax_bh, ay_bh, t_bh, error_bh)
    except Exception as e:
        print(f"   ✗ Barnes-Hut failed: {e}")
        return False
    
    # Test FMM method
    print("\n3. Testing FMM method...")
    try:
        t0 = time.time()
        ax_fmm, ay_fmm = fmm_omp(x, y, m, domain, theta, G, soft)
        t_fmm = time.time() - t0
        
        print(f"   ✓ FMM: {t_fmm:.4f} s")
        print(f"   Force range: ax=[{ax_fmm.min():.2e}, {ax_fmm.max():.2e}]")
        print(f"                ay=[{ay_fmm.min():.2e}, {ay_fmm.max():.2e}]")
        
        # Calculate error relative to direct
        error_fmm = np.mean(np.sqrt((ax_fmm - ax_direct)**2 + (ay_fmm - ay_direct)**2) / 
                           (np.sqrt(ax_direct**2 + ay_direct**2) + 1e-10))
        print(f"   Relative error vs Direct: {error_fmm:.2e}")
        results['fmm'] = (ax_fmm, ay_fmm, t_fmm, error_fmm)
    except Exception as e:
        print(f"   ✗ FMM failed: {e}")
        return False
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"Direct:     {results['direct'][2]:.4f} s")
    print(f"Barnes-Hut: {results['bh'][2]:.4f} s (error: {results['bh'][3]:.2e})")
    print(f"FMM:        {results['fmm'][2]:.4f} s (error: {results['fmm'][3]:.2e})")
    
    if results['bh'][3] < 0.1 and results['fmm'][3] < 0.1:
        print("✓ All methods working with acceptable accuracy!")
    else:
        print("⚠ Some methods have high errors - check implementation")
    
    # Create a simple comparison plot
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (method, data) in enumerate([('Direct', results['direct']), 
                                          ('Barnes-Hut', results['bh']), 
                                          ('FMM', results['fmm'])]):
            ax, ay = data[0], data[1]
            force_mag = np.sqrt(ax**2 + ay**2)
            
            scatter = axes[i].scatter(x, y, c=np.log10(force_mag + 1e-10), 
                                    s=m*2, cmap='viridis', alpha=0.7)
            axes[i].set_title(f'{method}\n({data[2]:.3f} s)')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('y')
            axes[i].set_aspect('equal')
            plt.colorbar(scatter, ax=axes[i], label='log10(|F|)')
        
        plt.tight_layout()
        plt.savefig('kernel_test_comparison.png', dpi=150)
        print("\n✓ Saved comparison plot as 'kernel_test_comparison.png'")
        plt.show()
        
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    return True

def test_energy_conservation():
    """Simple energy conservation test"""
    print("\n" + "="*50)
    print("Testing energy conservation...")
    
    try:
        from force_kernel import direct_omp
        
        # Create a simple 2-body system
        x = np.array([1.0, -1.0])
        y = np.array([0.0, 0.0]) 
        m = np.array([1.0, 1.0])
        vx = np.array([0.0, 0.0])
        vy = np.array([1.0, -1.0])
        
        dt = 0.01
        steps = 100
        
        energies = []
        
        for step in range(steps):
            # Calculate energy
            ke = 0.5 * np.sum(m * (vx**2 + vy**2))
            dx, dy = x[1] - x[0], y[1] - y[0]
            r = np.sqrt(dx**2 + dy**2 + 0.01**2)
            pe = -1.0 * m[0] * m[1] / r
            E = ke + pe
            energies.append(E)
            
            # Leapfrog integration
            ax, ay = direct_omp(x, y, m)
            
            # Half kick
            vx += ax * dt * 0.5
            vy += ay * dt * 0.5
            
            # Drift
            x += vx * dt
            y += vy * dt
            
            # Half kick
            ax, ay = direct_omp(x, y, m)
            vx += ax * dt * 0.5
            vy += ay * dt * 0.5
        
        energy_drift = abs(energies[-1] - energies[0]) / abs(energies[0])
        print(f"✓ Energy conservation test completed")
        print(f"   Initial energy: {energies[0]:.6f}")
        print(f"   Final energy:   {energies[-1]:.6f}")
        print(f"   Relative drift: {energy_drift:.2e}")
        
        if energy_drift < 0.01:
            print("✓ Good energy conservation!")
        else:
            print("⚠ Energy drift may be too large")
            
    except Exception as e:
        print(f"✗ Energy test failed: {e}")

if __name__ == "__main__":
    print("="*60)
    print("N-body Kernel Test Suite")
    print("="*60)
    
    if test_kernels():
        test_energy_conservation()
        print("\n" + "="*60)
        print("✓ ALL TESTS COMPLETED")
        print("You can now run the main program:")
        print("  python main_program_parallel_fixed.py")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ TESTS FAILED")
        print("Please check the compilation and try:")
        print("  python rebuild_and_test.py")
        print("="*60)
