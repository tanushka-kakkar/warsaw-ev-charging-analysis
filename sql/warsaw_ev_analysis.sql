-- ============================================================================
--  CAPSTONE PROJECT: Warsaw EV Charging Analytics
--  Part 1 - SQL Analysis (14 Queries)
--  Platform : MySQL 8.0.46
--  Schema   : warsaw_ev
--  Tables   : districts (18) | charging_stations (65)
--             customers (8000) | charging_sessions (114000)
-- ============================================================================

USE warsaw_ev;

-- ----------------------------------------------------------------------------
-- BASIC LEVEL
-- ----------------------------------------------------------------------------

-- Q1. List all districts along with their income tiers.
SELECT district_name,
       income_tier
FROM   districts
ORDER  BY income_tier, district_name;


-- Q2. Count the total number of charging stations in each district.
SELECT   d.district_name,
         d.income_tier,
         COUNT(st.station_id) AS total_stations
FROM     districts d
LEFT JOIN charging_stations st
       ON st.district_name = d.district_name
GROUP BY d.district_name, d.income_tier
ORDER  BY total_stations DESC, d.district_name;


-- Q3. Find the total number of charging plugs available per district.
SELECT   d.district_name,
         d.income_tier,
         COUNT(st.station_id)              AS total_stations,
         COALESCE(SUM(st.plugs_count), 0)  AS total_plugs
FROM     districts d
LEFT JOIN charging_stations st
       ON st.district_name = d.district_name
GROUP BY d.district_name, d.income_tier
ORDER  BY total_plugs DESC, d.district_name;


-- Q4. Calculate the total number of charging sessions recorded.
SELECT COUNT(*) AS total_charging_sessions
FROM   charging_sessions;


-- Q5. Find the average energy charged (kWh) per charging session.
SELECT ROUND(AVG(kwh_charged), 4) AS avg_kwh_per_session,
       ROUND(MIN(kwh_charged), 4) AS min_kwh,
       ROUND(MAX(kwh_charged), 4) AS max_kwh
FROM   charging_sessions;


-- ----------------------------------------------------------------------------
-- INTERMEDIATE LEVEL
-- ----------------------------------------------------------------------------

-- Q6. Calculate total energy charged (kWh) by each district.
SELECT   d.district_name,
         d.income_tier,
         COUNT(s.session_id)        AS total_sessions,
         ROUND(SUM(s.kwh_charged),2) AS total_kwh
FROM     districts d
JOIN     charging_stations st ON st.district_name = d.district_name
JOIN     charging_sessions  s  ON s.station_id    = st.station_id
GROUP BY d.district_name, d.income_tier
ORDER  BY total_kwh DESC;


-- Q7. Identify the top 5 charging stations based on number of charging sessions.
SELECT   st.station_id,
         st.district_name,
         st.operator_name,
         COUNT(s.session_id) AS total_sessions
FROM     charging_stations st
JOIN     charging_sessions  s ON s.station_id = st.station_id
GROUP BY st.station_id, st.district_name, st.operator_name
ORDER  BY total_sessions DESC, st.station_id
LIMIT  5;


-- Q8. Calculate total charging revenue generated per district.
SELECT   d.district_name,
         d.income_tier,
         COUNT(s.session_id)           AS total_sessions,
         ROUND(SUM(s.total_cost), 2)   AS total_revenue
FROM     districts d
JOIN     charging_stations st ON st.district_name = d.district_name
JOIN     charging_sessions  s  ON s.station_id    = st.station_id
GROUP BY d.district_name, d.income_tier
ORDER  BY total_revenue DESC;


-- Q9. Find the average charging cost per session for each income tier.
--     (income tier taken from the CUSTOMER, i.e. who is paying)
SELECT   c.income_tier,
         COUNT(DISTINCT c.customer_id) AS customers,
         COUNT(s.session_id)           AS total_sessions,
         ROUND(AVG(s.total_cost), 4)   AS avg_cost_per_session
FROM     customers c
JOIN     charging_sessions s ON s.customer_id = c.customer_id
GROUP BY c.income_tier
ORDER  BY avg_cost_per_session DESC;


-- Q10. Analyze average energy charged per customer.
SELECT ROUND(AVG(customer_total_kwh), 4) AS avg_kwh_per_customer,
       ROUND(MIN(customer_total_kwh), 4) AS min_kwh_per_customer,
       ROUND(MAX(customer_total_kwh), 4) AS max_kwh_per_customer
FROM  (SELECT customer_id,
              SUM(kwh_charged) AS customer_total_kwh
       FROM   charging_sessions
       GROUP  BY customer_id) t;


-- ----------------------------------------------------------------------------
-- ADVANCED LEVEL
-- ----------------------------------------------------------------------------

-- Q11. Compare projected EV count with total available charging plugs
--      for each district.
SELECT   d.district_name,
         d.income_tier,
         d.projected_evs,
         d.projected_plugs                                   AS planned_plugs,
         COALESCE(SUM(st.plugs_count), 0)                    AS available_plugs,
         COUNT(st.station_id)                                AS stations,
         ROUND(d.projected_evs /
               NULLIF(COALESCE(SUM(st.plugs_count),0),0), 2) AS projected_evs_per_plug
FROM     districts d
LEFT JOIN charging_stations st
       ON st.district_name = d.district_name
GROUP BY d.district_name, d.income_tier, d.projected_evs, d.projected_plugs
ORDER  BY d.projected_evs DESC;


-- Q12. Identify districts where projected EVs exceed available charging plugs.
SELECT   d.district_name,
         d.income_tier,
         d.projected_evs,
         COALESCE(SUM(st.plugs_count), 0) AS available_plugs,
         d.projected_evs -
           COALESCE(SUM(st.plugs_count), 0) AS plug_shortfall
FROM     districts d
LEFT JOIN charging_stations st
       ON st.district_name = d.district_name
GROUP BY d.district_name, d.income_tier, d.projected_evs
HAVING   d.projected_evs > COALESCE(SUM(st.plugs_count), 0)
ORDER  BY plug_shortfall DESC;


-- Q13. Analyze charging demand per projected EV by district
--      (total kWh / projected EVs).
SELECT   d.district_name,
         d.projected_evs,
         COALESCE(x.total_kwh, 0)                          AS actual_kwh,
         ROUND(COALESCE(x.total_kwh, 0) /
               d.projected_evs, 4)                         AS kwh_per_projected_ev
FROM     districts d
LEFT JOIN (SELECT st.district_name,
                  SUM(s.kwh_charged) AS total_kwh
           FROM   charging_stations st
           JOIN   charging_sessions  s ON s.station_id = st.station_id
           GROUP  BY st.district_name) x
       ON x.district_name = d.district_name
ORDER  BY kwh_per_projected_ev DESC;


-- Q14. Rank districts based on infrastructure readiness using actual
--      charging usage vs projections.
--
--  Readiness Index (documented, reproducible):
--      plug_plan_coverage_pct = (available_plugs / projected_plugs) * 100
--                               -> SUPPLY side: how much of the planned
--                                  plug network is actually built.
--      actual_kwh_per_ev      = actual_kwh / projected_evs
--                               -> DEMAND side: realised load per planned EV.
--
--      readiness_rank = RANK() OVER (
--                          ORDER BY plug_plan_coverage_pct DESC,  -- more supply = better
--                                    actual_kwh_per_ev      ASC   -- less strain = better
--                       )
--      Rank 1 = BEST prepared, Rank 18 = WORST prepared.
WITH station_agg AS (
    SELECT st.station_id,
           st.district_name,
           st.plugs_count,
           COUNT(s.session_id)             AS sessions,
           COALESCE(SUM(s.kwh_charged), 0) AS kwh
    FROM   charging_stations st
    LEFT JOIN charging_sessions s
           ON s.station_id = st.station_id
    GROUP  BY st.station_id, st.district_name, st.plugs_count
),
district_agg AS (
    SELECT   d.district_name,
             d.income_tier,
             d.projected_evs,
             d.projected_plugs,
             COALESCE(SUM(sa.plugs_count), 0) AS available_plugs,
             COALESCE(SUM(sa.sessions), 0)    AS actual_sessions,
             COALESCE(SUM(sa.kwh), 0)         AS actual_kwh
    FROM     districts d
    LEFT JOIN station_agg sa
           ON sa.district_name = d.district_name
    GROUP BY d.district_name, d.income_tier,
             d.projected_evs, d.projected_plugs
),
scored AS (
    SELECT district_name,
           income_tier,
           projected_evs,
           projected_plugs,
           available_plugs,
           actual_sessions,
           actual_kwh,
           ROUND(available_plugs / projected_plugs * 100, 2) AS plug_plan_coverage_pct,
           ROUND(actual_kwh / projected_evs, 2)              AS actual_kwh_per_ev,
           ROUND(actual_sessions / projected_evs, 4)         AS sessions_per_ev
    FROM   district_agg
)
SELECT district_name,
       income_tier,
       projected_evs,
       available_plugs,
       plug_plan_coverage_pct,
       actual_kwh,
       actual_kwh_per_ev,
       sessions_per_ev,
       RANK() OVER (ORDER BY plug_plan_coverage_pct DESC,
                             actual_kwh_per_ev      ASC) AS readiness_rank
FROM   scored
ORDER  BY readiness_rank, district_name;
