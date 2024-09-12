# HTB-TheLastDance_BruteForcer

A Proof of Concept Brute Forcer for HTB's "The Last Dance" Challenge

This repository contains a proof of concept brute forcer for the "The Last Dance" challenge on Hack The Box. The challenge can be found [here](https://app.hackthebox.com/challenges/The%2520Last%2520Dance).

## Overview

This project demonstrates an attempt to leverage GPU CUDA acceleration to brute force the challenge key. The current implementation achieves a processing rate of up to 1 million keys per second.

**Important Note:** Despite the high processing speed, using this tool to calculate the correct key would still require an impractical amount of time, making it unfeasible for practical use.

## Purpose

This project is intended purely as a proof of concept, showcasing the potential of GPU acceleration in brute force scenarios. While not viable for real-world application due to the immense time required, it stands as an interesting technical demonstration.

## Disclaimer

This code is provided "as-is" and is intended for educational and research purposes only. The author does not endorse using this tool for any illegal activities.

## Conclusion

While the brute forcer is not a viable solution for cracking the challenge key within a reasonable timeframe, it serves as a neat piece of code that explores the use of GPU acceleration in cryptographic brute force attacks.
