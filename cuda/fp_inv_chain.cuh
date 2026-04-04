// BLS12-381 Fp inversion via addition chain
// Ported from blst/src/recip-addchain.h (Apache-2.0)
// Generated with 'addchain' tool by kwantam
// 461 squarings + 67 multiplications (vs ~570 ops for naive Fermat)

#pragma once

// Requires fp_sqr and fp_mul to be defined

__device__ __forceinline__ Fp sqr_n_mul(Fp a, int n, const Fp& b) {
    for (int i = 0; i < n; i++) a = fp_sqr(a);
    return fp_mul(a, b);
}

__device__ __noinline__ Fp fp_inv_chain(const Fp& inp) {
    Fp t[16];
    t[1] = inp;
    t[0] = fp_sqr(t[1]);                   //  1: 2
    t[9] = fp_mul(t[0], t[1]);             //  2: 3
    t[5] = fp_sqr(t[0]);                   //  3: 4
    t[2] = fp_mul(t[9], t[0]);             //  4: 5
    t[7] = fp_mul(t[5], t[9]);             //  5: 7
    t[10]= fp_mul(t[2], t[5]);             //  6: 9
    t[13]= fp_mul(t[7], t[5]);             //  7: b
    t[4] = fp_mul(t[10], t[5]);            //  8: d
    t[8] = fp_mul(t[13], t[5]);            //  9: f
    t[15]= fp_mul(t[4], t[5]);             // 10: 11
    t[11]= fp_mul(t[8], t[5]);             // 11: 13
    t[3] = fp_mul(t[15], t[5]);            // 12: 15
    t[12]= fp_mul(t[11], t[5]);            // 13: 17
    t[0] = fp_sqr(t[4]);                   // 14: 1a
    t[14]= fp_mul(t[12], t[5]);            // 15: 1b
    t[6] = fp_mul(t[0], t[9]);             // 16: 1d
    t[5] = fp_mul(t[0], t[2]);             // 17: 1f

    t[0] = sqr_n_mul(t[0], 12, t[15]);     // 30: 1a011
    t[0] = sqr_n_mul(t[0], 7, t[8]);       // 38: d0088f
    t[0] = sqr_n_mul(t[0], 4, t[2]);       // 43: d0088f5
    t[0] = sqr_n_mul(t[0], 6, t[7]);       // 50: 340223d47
    t[0] = sqr_n_mul(t[0], 7, t[12]);      // 58: 1a0111ea397
    t[0] = sqr_n_mul(t[0], 5, t[5]);       // 64: 340223d472ff
    t[0] = sqr_n_mul(t[0], 2, t[9]);       // 67: d0088f51cbff
    t[0] = sqr_n_mul(t[0], 6, t[4]);       // 74: 340223d472ffcd
    t[0] = sqr_n_mul(t[0], 6, t[4]);       // 81: d0088f51cbff34d
    t[0] = sqr_n_mul(t[0], 6, t[10]);      // 88: 340223d472ffcd349
    t[0] = sqr_n_mul(t[0], 3, t[9]);       // 92: 1a0111ea397fe69a4b
    t[0] = sqr_n_mul(t[0], 7, t[4]);       //100: d0088f51cbff34d258d
    t[0] = sqr_n_mul(t[0], 4, t[4]);       //105: d0088f51cbff34d258dd
    t[0] = sqr_n_mul(t[0], 6, t[8]);       //112: 340223d472ffcd3496374f
    t[0] = sqr_n_mul(t[0], 6, t[14]);      //119: d0088f51cbff34d258dd3db
    t[0] = sqr_n_mul(t[0], 3, t[1]);       //123: 680447a8e5ff9a692c6e9ed9
    t[0] = sqr_n_mul(t[0], 8, t[4]);       //132: 680447a8e5ff9a692c6e9ed90d
    t[0] = sqr_n_mul(t[0], 7, t[12]);      //140: 340223d472ffcd3496374f6c8697
    t[0] = sqr_n_mul(t[0], 5, t[13]);      //146: 680447a8e5ff9a692c6e9ed90d2eb
    t[0] = sqr_n_mul(t[0], 6, t[4]);       //153: d0088f51cbff34d258dd3db21a5d6d
    t[0] = sqr_n_mul(t[0], 6, t[7]);       //160: 340223d472ffcd3496374f6c869758747
    t[0] = sqr_n_mul(t[0], 3, t[9]);       //164: 1a0111ea397fe69a4b1ba7b6434bacd764b
    t[0] = sqr_n_mul(t[0], 7, t[3]);       //172: d0088f51cbff34d258dd3db21a5d66bb2595
    t[0] = sqr_n_mul(t[0], 4, t[1]);       //177: d0088f51cbff34d258dd3db21a5d66bb25951
    t[0] = sqr_n_mul(t[0], 4, t[9]);       //182: d0088f51cbff34d258dd3db21a5d66bb259513
    t[0] = sqr_n_mul(t[0], 5, t[2]);       //188: 1a0111ea397fe69a4b1ba7b6434bacd764b2a265
    t[0] = sqr_n_mul(t[0], 7, t[2]);       //196: d0088f51cbff34d258dd3db21a5d66bb2595132c5
    t[0] = sqr_n_mul(t[0], 3, t[9]);       //200: 680447a8e5ff9a692c6e9ed90d2eb35d92ca89962b
    t[0] = sqr_n_mul(t[0], 6, t[3]);       //207: 1a0111ea397fe69a4b1ba7b6434bacd764b2a265895
    t[0] = sqr_n_mul(t[0], 5, t[2]);       //213: 340223d472ffcd3496374f6c869759ec965944cb12a5
    t[0] = sqr_n_mul(t[0], 5, t[10]);      //219: d0088f51cbff34d258dd3db21a5d67b259651932c4a949
    t[0] = sqr_n_mul(t[0], 3, t[9]);       //223: 680447a8e5ff9a692c6e9ed90d2eb3d92cb28c99625a4ab
    t[0] = sqr_n_mul(t[0], 7, t[3]);       //231: 340223d472ffcd3496374f6c869759ec965944cb12d25295
    t[0] = sqr_n_mul(t[0], 5, t[10]);      //237: d0088f51cbff34d258dd3db21a5d67b259651932c4b494a549
    t[0] = sqr_n_mul(t[0], 4, t[13]);      //242: d0088f51cbff34d258dd3db21a5d67b259651932c4b494a54ab
    t[0] = sqr_n_mul(t[0], 4, t[2]);       //247: d0088f51cbff34d258dd3db21a5d67b259651932c4b494a54ab5
    t[0] = sqr_n_mul(t[0], 8, t[6]);       //256: d0088f51cbff34d258dd3db21a5d67b259651932c4b494a54ab51d
    t[0] = sqr_n_mul(t[0], 5, t[2]);       //262: 1a0111ea397fe69a4b1ba7b6434bacd764b2a32658969293a956a3a5
    t[0] = sqr_n_mul(t[0], 5, t[10]);      //268: 680447a8e5ff9a692c6e9ed90d2eb35d92ca8c996258a4a4ea55a8e949
    t[0] = sqr_n_mul(t[0], 4, t[10]);      //273: 680447a8e5ff9a692c6e9ed90d2eb35d92ca8c996258a4a4ea55a8e9499
    t[0] = sqr_n_mul(t[0], 6, t[4]);       //280: 1a0111ea397fe69a4b1ba7b6434bacd764b2a32658969293a956a23a52664d
    t[0] = sqr_n_mul(t[0], 6, t[11]);      //287: 680447a8e5ff9a692c6e9ed90d2eb35d92ca8c996258a4a4ea55a88e499934d3
    t[0] = sqr_n_mul(t[0], 4, t[7]);       //292: 680447a8e5ff9a692c6e9ed90d2eb35d92ca8c996258a4a4ea55a88e499934d37
    t[0] = sqr_n_mul(t[0], 6, t[15]);      //299: 1a0111ea397fe69a4b1ba7b6434bacd764b2a32658969293a956a23a5266135134c11
    t[0] = sqr_n_mul(t[0], 5, t[2]);       //305: 340223d472ffcd3496374f6c869759ec965946530b2d25274d2ad4474a4c26a2698225
    t[0] = sqr_n_mul(t[0], 5, t[10]);      //311: d0088f51cbff34d258dd3db21a5d67b259651594c2cb494d34ab510d2930989a4a608949
    t[0] = sqr_n_mul(t[0], 2, t[1]);       //314: 680447a8e5ff9a692c6e9ed90d2eb3d92ca8aca6165a4a69a5558868949804cd253044a49
    t[0] = sqr_n_mul(t[0], 6, t[4]);       //321: 1a0111ea397fe69a4b1ba7b6434bacd764b2a2b298169293a6955e21a25260133494c11292d
    t[0] = sqr_n_mul(t[0], 5, t[8]);       //327: 340223d472ffcd3496374f6c869759ec96545654302d25274d2abc4344a4c02669298224525af
    t[0] = sqr_n_mul(t[0], 6, t[3]);       //334: d0088f51cbff34d258dd3db21a5d67b259515950c0b494d34ab2f10d2930009a4a60891494ebc15
    t[0] = sqr_n_mul(t[0], 5, t[10]);      //340: 1a0111ea397fe69a4b1ba7b6434bacd764b2a2b2a18169293a965e21a5260013494c1122929d7829
    t[0] = sqr_n_mul(t[0], 6, t[11]);      //347: 680447a8e5ff9a692c6e9ed90d2eb35d92ca8aca860e5a4a4ea59788694980004d253044a4a75e0a4d3
    t[0] = sqr_n_mul(t[0], 5, t[8]);       //353: d0088f51cbff34d258dd3db21a5d67b25959159580c1cb494d34b2f10d2930000269a60891494ebc149a6f
    t[0] = sqr_n_mul(t[0], 5, t[2]);       //359: 1a0111ea397fe69a4b1ba7b6434bacd764b2a2b2b01839693a6965e21a52600004d34c11229297829351de5
    t[0] = sqr_n_mul(t[0], 5, t[7]);       //365: 340223d472ffcd3496374f6c869759ec965946565603072d274d2cbc4344a4c000009a698224522f05270a3bc7
    t[0] = sqr_n_mul(t[0], 3, t[9]);       //369: 1a0111ea397fe69a4b1ba7b6434bacd764b2a2b2ab01839693a6965e21a52600004d34c1122929f8292938515e3b
    t[0] = sqr_n_mul(t[0], 8, t[3]);       //378: 1a0111ea397fe69a4b1ba7b6434bacd764b2a2b2ab01839693a6965e21a52600004d34c1122929f8292938515e3b15
    t[0] = sqr_n_mul(t[0], 6, t[13]);      //385: 680447a8e5ff9a692c6e9ed90d2eb35d92ca8acacc060e5a4e9a597884694980001534d30448a4a7e0a4a4e14578ec54b
    t[0] = sqr_n_mul(t[0], 4, t[2]);       //390: 680447a8e5ff9a692c6e9ed90d2eb35d92ca8acacc060e5a4e9a597884694980001534d30448a4a7e0a4a4e14578ec54b5
    t[0] = sqr_n_mul(t[0], 6, t[4]);       //397: 1a0111ea397fe69a4b1ba7b6434bacd764b2a2b2b3018396939a695e21a126002600054d34c1122929f8292938515e3b152d
    t[0] = sqr_n_mul(t[0], 7, t[6]);       //405: d0088f51cbff34d258dd3db21a5d67b259515959980c1cb49cd34af10d0930001300002a69a60891494fc149494a0af1d8a969d
    t[0] = sqr_n_mul(t[0], 5, t[11]);      //411: 1a0111ea397fe69a4b1ba7b6434bacd764b2a2b2b30183969399695e21a1260002600005534c112292939f82929385152e3b152d3a13
    t[0] = sqr_n_mul(t[0], 4, t[10]);      //416: 1a0111ea397fe69a4b1ba7b6434bacd764b2a2b2b30183969399695e21a1260002600005534c112292939f82929385152e3b152d3a139
    t[0] = sqr_n_mul(t[0], 4, t[7]);       //421: 1a0111ea397fe69a4b1ba7b6434bacd764b2a2b2b30183969399695e21a1260002600005534c112292939f82929385152e3b152d3a1397
    t[0] = sqr_n_mul(t[0], 6, t[4]);       //428: 680447a8e5ff9a692c6e9ed90d2eb35d92ca8acaccc060e5a4e669578868498000980001554d30448a4a4e7e0a4a4e14454b8ec54b4e84e5d
    t[0] = sqr_n_mul(t[0], 5, t[15]);      //434: d0088f51cbff34d258dd3db21a5d67b25959159598180c1cb49ccd2af10d0930001300002aa9a60891494d9fc14949c2888971d8a969d09cabd11
    t[0] = sqr_n_mul(t[0], 4, t[2]);       //439: d0088f51cbff34d258dd3db21a5d67b25959159598180c1cb49ccd2af10d0930001300002aa9a60891494d9fc14949c2888971d8a969d09cabd115
    t[0] = sqr_n_mul(t[0], 6, t[12]);      //446: 340223d472ffcd3496374f6c869759ec965645654606030072d274334abc4343424c00004aaa698224252536f7052527420a225c762a5a7342742f4c4557
    t[0] = sqr_n_mul(t[0], 6, t[11]);      //453: d0088f51cbff34d258dd3db21a5d67b259515959181b0c00cb49d0cd2af10d0d0930000012aa9a6089094d4ddbdc1494d908289571e8a969cd09d0bd31315d3
    t[0] = sqr_n_mul(t[0], 4, t[9]);       //458: d0088f51cbff34d258dd3db21a5d67b259515959181b0c00cb49d0cd2af10d0d0930000012aa9a6089094d4ddbdc1494d908289571e8a969cd09d0bd31315d33
    t[0] = sqr_n_mul(t[0], 6, t[8]);       //465: 340223d472ffcd3496374f6c869759ec9654565464606c30032d274334abc4343424c00004aaa6989824253535376f05253642420a255c7a2a5a734272742f4c4574ccf

    // Final: the result is (p-2) encoded in the addition chain
    // vec_copy(out, t[0], sizeof(ptype));
    return t[0];
}
