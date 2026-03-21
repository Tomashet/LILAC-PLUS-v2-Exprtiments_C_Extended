$envs = "highway-v0"
$constraints = "none","cpss","proactive_forecast","adjust_speed"
$pstay = 1.0,0.95,0.85,0.7
$seeds = 0,1,2,3,4

foreach ($c in $constraints) {
 foreach ($p in $pstay) {
  foreach ($s in $seeds) {

   python scripts/train_continuous.py `
    --env highway-v0 `
    --constraint $c `
    --p_stay $p `
    --seed $s `
    --total_steps 500000

  }
 }
}