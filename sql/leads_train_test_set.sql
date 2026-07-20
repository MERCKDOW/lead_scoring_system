# current table

CREATE OR REPLACE TABLE `expanded-nebula-754`.`sandbox_crdow`.`leads_training_set_text_category_IID`
AS
WITH
  golden_data AS (
    SELECT
      std_email_address,
      individual_id,
      ARRAY_AGG(DISTINCT search_term IGNORE NULLS) AS a_search_terms,
      ARRAY_AGG(DISTINCT source_campaign IGNORE NULLS) AS a_source_campaigns,
      ARRAY_AGG(DISTINCT department IGNORE NULLS) AS a_departments,
      ARRAY_AGG(DISTINCT sfdc_product_of_greatest_interest_pogi IGNORE NULLS)
        AS a_product_interest,
      ARRAY_AGG(DISTINCT lead_source_most_recent IGNORE NULLS)
        AS a_leads_recent,
      ARRAY_AGG(DISTINCT nielsen_id_lvl_1 IGNORE NULLS) AS a_nielsen_id_lvl_1,
      ARRAY_AGG(DISTINCT nielsen_id_lvl_2 IGNORE NULLS) AS a_nielsen_id_lvl_2,
      SUM(IFNULL(total_pageviews_count, 0)) AS page_views,
      -- Convert last_order_date to days since current date
      AVG(
        CASE
          WHEN last_order_date IS NOT NULL
            THEN UNIX_DATE(CURRENT_DATE()) - UNIX_DATE(DATE(last_order_date))
          ELSE NULL
          END)
        AS days_since_last_order,
      -- Convert last_visit to days since current date
      AVG(
        CASE
          WHEN last_visit IS NOT NULL
            THEN UNIX_DATE(CURRENT_DATE()) - UNIX_DATE(DATE(last_visit))
          ELSE NULL
          END)
        AS days_since_last_visit,
      AVG(avg_session_duration) AS avg_session_duration
    FROM `expanded-nebula-754.sanbox_cdow.golden_records`
    WHERE std_email_address IS NOT NULL
    GROUP BY individual_id, std_email_address
  ),
  clicks_per_email AS (
    SELECT subscriberkey, ARRAY_AGG(DISTINCT url IGNORE NULLS) AS urls
    FROM `expanded-nebula-754.SFMC.sfmc_click`
    WHERE subscriberkey IS NOT NULL
    GROUP BY subscriberkey
  ),
  sfmc_job AS (
    SELECT jobid, emailsubject, iswrapped
    FROM `expanded-nebula-754.SFMC.sfmc_job`
    WHERE jobid IS NOT NULL
  ),
  sfmc_sent AS (
    SELECT jobid, subscriberkey,
    FROM `expanded-nebula-754.SFMC.sfmc_sent`
    GROUP BY jobid, subscriberkey
  ),
  sfmc_email_text AS (
    SELECT t2.subscriberkey, ARRAY_AGG(t1.emailsubject) AS a_emailsubject,
    FROM sfmc_sent AS t2
    INNER JOIN sfmc_job AS t1
      ON t1.jobid = t2.jobid
    GROUP BY t2.subscriberkey
  ),
  golden_record_set AS (
    SELECT
      gd.std_email_address,
      gd.Individual_id,
      gd.a_search_terms,
      gd.a_source_campaigns,
      gd.a_product_interest,
      gd.a_leads_recent,
      gd.a_nielsen_id_lvl_1,
      gd.a_nielsen_id_lvl_2,
      gd.page_views,
      gd.days_since_last_order,
      gd.days_since_last_visit,
      gd.avg_session_duration,
      cp.urls AS a_url
    FROM golden_data gd
    LEFT JOIN clicks_per_email cp
      ON cp.subscriberkey = gd.std_email_address
  ),
  golden_record_click_text_set AS (
    SELECT
      grs.std_email_address,
      grs.Individual_id,
      grs.a_search_terms,
      grs.a_source_campaigns,
      grs.a_product_interest,
      grs.a_leads_recent,
      grs.a_nielsen_id_lvl_1,
      grs.a_nielsen_id_lvl_2,
      grs.page_views,
      grs.days_since_last_order,
      grs.days_since_last_visit,
      grs.avg_session_duration,
      grs.a_url,
      et.a_emailsubject
    FROM golden_record_set AS grs
    LEFT JOIN sfmc_email_text et
      ON grs.std_email_address = et.subscriberkey
  ),
  leads AS (
    SELECT
      l.ConvertedOpportunityId AS ConvertedOpportunityId, l.Email AS a_email
    FROM `expanded-nebula-754.SFDC.lead` l
  ),
  labelled_opportunities AS (
    SELECT o.opportunity_id, o.is_won AS a_is_won,
    FROM `expanded-nebula-754.SFDC.opportunities_as` o
    WHERE o.is_won IS NOT NULL
  ),
  leads_with_won_status AS (
    SELECT
      l.a_email AS lead_email,
      MAX(lo.a_is_won)
        AS is_won_status  -- Aggregate if multiple opportunities per email
    FROM leads l
    LEFT JOIN labelled_opportunities lo
      ON l.ConvertedOpportunityId = lo.opportunity_id
    WHERE l.a_email IS NOT NULL  -- Ensure we only consider emails that exist
    GROUP BY l.a_email
  ),
  filtered_grcts AS (
    SELECT
      grcts.* EXCEPT (std_email_address),
      grcts.std_email_address,
      lws.is_won_status
    FROM golden_record_click_text_set grcts
    LEFT JOIN leads_with_won_status lws
      ON grcts.std_email_address = lws.lead_email
    WHERE
      NOT (
        (grcts.a_search_terms IS NULL OR ARRAY_LENGTH(grcts.a_search_terms) = 0)
        AND (
          grcts.a_source_campaigns IS NULL
          OR ARRAY_LENGTH(grcts.a_source_campaigns) = 0)
        AND (
          grcts.a_product_interest IS NULL
          OR ARRAY_LENGTH(grcts.a_product_interest) = 0)
        AND (
          grcts.a_leads_recent IS NULL
          OR ARRAY_LENGTH(grcts.a_leads_recent) = 0)
        AND (
          grcts.a_nielsen_id_lvl_1 IS NULL
          OR ARRAY_LENGTH(grcts.a_nielsen_id_lvl_1) = 0)
        AND (
          grcts.a_nielsen_id_lvl_2 IS NULL
          OR ARRAY_LENGTH(grcts.a_nielsen_id_lvl_2) = 0)
        AND (grcts.page_views IS NULL OR grcts.page_views = 0)
        AND (grcts.days_since_last_order IS NULL)
        AND (grcts.days_since_last_visit IS NULL)
        AND (
          grcts.avg_session_duration IS NULL OR grcts.avg_session_duration = 0)
        AND (grcts.a_url IS NULL OR ARRAY_LENGTH(grcts.a_url) = 0)
        AND (
          grcts.a_emailsubject IS NULL
          OR ARRAY_LENGTH(grcts.a_emailsubject) = 0))
  )
-- Combine labelled data with a limited sample of unlabelled data
SELECT *
FROM filtered_grcts
WHERE is_won_status IS NOT NULL
UNION ALL
(
  SELECT * FROM filtered_grcts WHERE is_won_status IS NULL LIMIT 10000
)  -- Replace 1000 with your desired N

