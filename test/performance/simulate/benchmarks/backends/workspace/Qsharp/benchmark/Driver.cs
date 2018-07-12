using Microsoft.Quantum.Simulation.Core;
using Microsoft.Quantum.Simulation.Simulators;
using System;

namespace QasmBench.Bench
{
    class Driver
    {
        static void Main(string[] args)
        {
          var sw = new System.Diagnostics.Stopwatch();
          using (var sim = new QuantumSimulator())
          {
            sw.Start();
            // Try initial values
            var res = Bench.Run(sim).Result.ToArray();
            /*for(int i = 0; i < res.Length; i++){
              Console.Write($"State: {res[i]} \n");
              }*/
            //Console.Write($"State: {string.Join(", ", res.Select(x=>x.ToString()).ToArray())} \n");
          }
          sw.Stop();
          TimeSpan ts = sw.Elapsed;
          //Console.WriteLine($"  {ts}");
          //Console.WriteLine($"  {ts.Hours} h {ts.Minutes} m {ts.Seconds} sec {ts.Milliseconds} mil");
          Console.WriteLine($"  {sw.ElapsedMilliseconds/1000}.{ts.Milliseconds}");

        }
    }
}
