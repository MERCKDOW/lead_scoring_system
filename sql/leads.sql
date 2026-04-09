# current table

CREATE OR REPLACE TABLE `expanded-nebula-754`.`sandbox_crdow`.`leads_training_set_text_category`
AS
WITH
  golden_data AS (
    SELECT
      std_email_address,
      ARRAY_AGG(DISTINCT search_term IGNORE NULLS) AS a_search_terms,
      ARRAY_AGG(DISTINCT source_campaign IGNORE NULLS) AS a_source_campaigns,
      ARRAY_AGG(DISTINCT department IGNORE NULLS) AS a_departments,
      ARRAY_AGG(DISTINCT sfdc_product_of_greatest_interest_pogi IGNORE NULLS)
        AS a_product_interest,
      ARRAY_AGG(DISTINCT lead_source_most_recent IGNORE NULLS)
        AS a_leads_recent,
      ARRAY_AGG(DISTINCT nielsen_id_lvl_1 IGNORE NULLS)
        AS a_nielsen_id_lvl_1,        
      ARRAY_AGG(DISTINCT nielsen_id_lvl_2 IGNORE NULLS)
        AS a_nielsen_id_lvl_2,        
      SUM(IFNULL(total_pageviews_count, 0)) AS page_views

    FROM `expanded-nebula-754.sanbox_cdow.golden_records`
    WHERE std_email_address IS NOT NULL
    GROUP BY std_email_address
  ),
  leads_enriched AS (
    SELECT
      ConvertedOpportunityId,
      ARRAY_AGG(DISTINCT l.Email IGNORE NULLS) AS a_email,
      ARRAY_AGG(DISTINCT l.CurrencyIsoCode IGNORE NULLS) AS a_currency,
      ARRAY_CONCAT_AGG(re.a_search_terms) AS a_search_terms,
      ARRAY_CONCAT_AGG(re.a_source_campaigns) AS a_source_campaigns,
      ARRAY_CONCAT_AGG(re.a_product_interest) AS a_product_interest,
      ARRAY_CONCAT_AGG(re.a_leads_recent) AS a_leads_recent,
      ARRAY_CONCAT_AGG(re.a_nielsen_id_lvl_1) AS a_nielsen_id_lvl_1,
      ARRAY_CONCAT_AGG(re.a_nielsen_id_lvl_2) AS a_nielsen_id_lvl_2,

      #SUM(IFNULL(l.Days_Open__c, 0)) AS days_open,
      SUM(IFNULL(re.page_views, 0)) AS page_views

    FROM `expanded-nebula-754.SFDC.lead` l
    LEFT JOIN golden_data re
      ON l.Email = re.std_email_address
    WHERE l.Email IS NOT NULL AND ConvertedAccountId IS NOT NULL
    GROUP BY ConvertedOpportunityId
  ),
  clicks_per_email AS (
    SELECT subscriberkey, ARRAY_AGG(DISTINCT url IGNORE NULLS) AS urls
    FROM `expanded-nebula-754.SFMC.sfmc_click`
    WHERE subscriberkey IS NOT NULL
    GROUP BY subscriberkey
  ),
  -- sfmc_job AS (SELECT jobid, ARRAY_AGG(DISTINCT emailsubject IGNORE NULLS) AS a_emailsubject, ARRAY_AGG(DISTINCT iswrapped IGNORE NULLS)
  --                    AS a_iswrapped
  --             FROM `expanded-nebula-754.SFMC.sfmc_job`
  --             WHERE jobid IS NOT NULL
  --             GROUP BY jobid
  --             ),
  sfmc_job AS (
    SELECT jobid, emailsubject, iswrapped
    FROM `expanded-nebula-754.SFMC.sfmc_job`
    WHERE jobid IS NOT NULL
  ),
  -- sfmc_sent AS (SELECT jobid, ARRAY_AGG(DISTINCT subscriberkey IGNORE NULLS) AS a_subscriberkey,
  --              FROM `expanded-nebula-754.SFMC.sfmc_sent`
  --              GROUP BY jobid),
  sfmc_sent AS (
    SELECT jobid, subscriberkey,
    FROM `expanded-nebula-754.SFMC.sfmc_sent`
    GROUP BY jobid, subscriberkey
  ),
  -- sfmc_email_text AS (SELECT t1.jobid, ARRAY_CONCAT_AGG(t2.a_subscriberkey) AS a_subscriberkey, ARRAY_CONCAT_AGG(t1.a_emailsubject) AS a_emailsubject,
  --                    FROM sfmc_job as t1
  --                    LEFT JOIN sfmc_sent as t2
  --                    ON t1.jobid = t2.jobid
  --                    GROUP BY t1.jobid),
  sfmc_email_text AS (
    SELECT t2.subscriberkey, ARRAY_AGG(t1.emailsubject) AS a_emailsubject,
    FROM sfmc_sent AS t2
    INNER JOIN sfmc_job AS t1
      ON t1.jobid = t2.jobid
    GROUP BY t2.subscriberkey
  ),
  labelled_opportunities_click AS (
    SELECT
      o.opportunity_id,
      ARRAY_AGG(DISTINCT email IGNORE NULLS) AS a_email,
      ARRAY_CONCAT_AGG(le.a_search_terms) AS a_search_terms,
      ARRAY_CONCAT_AGG(le.a_source_campaigns) AS a_source_campaigns,
      ARRAY_CONCAT_AGG(le.a_currency) AS a_currency,
      ARRAY_CONCAT_AGG(le.a_product_interest) AS a_product_interest,
      ARRAY_CONCAT_AGG(le.a_leads_recent) AS a_leads_recent,
      ARRAY_CONCAT_AGG(le.a_nielsen_id_lvl_1) AS a_nielsen_id_lvl_1,
      ARRAY_CONCAT_AGG(le.a_nielsen_id_lvl_2) AS a_nielsen_id_lvl_2,
      ARRAY_CONCAT_AGG(cp.urls) AS a_url,
      ARRAY_AGG(DISTINCT o.recordtypename_c IGNORE NULLS) AS a_recordtypename_c,
      ARRAY_AGG(DISTINCT o.is_won IGNORE NULLS) AS a_is_won,
      ARRAY_AGG(DISTINCT o.is_closed IGNORE NULLS) AS a_is_closed,
      ARRAY_AGG(DISTINCT o.Type IGNORE NULLS) AS a_Type,
      ARRAY_AGG(DISTINCT o.campaign_name IGNORE NULLS) AS a_campaign_name,
      SUM(IFNULL(le.page_views, 0)) AS page_views,
      #SUM(IFNULL(le.days_open, 0)) AS days_open
    FROM `expanded-nebula-754.SFDC.opportunities_as` o
    LEFT JOIN leads_enriched le
      ON o.opportunity_id = le.ConvertedOpportunityId
    LEFT JOIN UNNEST(le.a_email) AS email
    LEFT JOIN clicks_per_email cp
      ON cp.subscriberkey = email
    WHERE le.a_email IS NOT NULL AND o.opportunity_id IS NOT NULL
    GROUP BY o.opportunity_id
  ),
  -- labeled_opportunities_click_text AS (SELECT loc.*, ARRAY_CONCAT_AGG(et.a_emailsubject) AS a_emailsubject
  --                                     FROM labelled_opportunities_click loc
  --                                     LEFT JOIN UNNEST(loc.a_email) AS email
  --                                     LEFT JOIN sfmc_email_text et
  --                                     ON email IN UNNEST(et.a_subscriberkey)
  --                                     GROUP BY loc.opportunity_id, loc.a_email,a_currency, loc.a_search_terms, loc.a_source_campaigns, loc.a_product_interest, loc.a_leads_recent, loc.a_url, loc.a_is_won,
  --                                              loc.a_is_closed, loc.a_Type, loc.a_campaign_name, loc.page_views,loc.days_open,loc.a_recordtypename_c)
  labeled_opportunities_click_text AS (
    SELECT loc.*, ARRAY_CONCAT_AGG(et.a_emailsubject) AS a_emailsubject
    FROM labelled_opportunities_click loc
    LEFT JOIN UNNEST(loc.a_email) AS email
    LEFT JOIN sfmc_email_text et
      ON email = et.subscriberkey
    GROUP BY
      loc.opportunity_id, loc.a_email, a_currency, loc.a_search_terms,
      loc.a_source_campaigns, loc.a_product_interest, loc.a_leads_recent,
      loc.a_url, loc.a_is_won, loc.a_is_closed, loc.a_Type, loc.a_campaign_name,
      loc.page_views,loc.a_nielsen_id_lvl_1,loc.a_nielsen_id_lvl_2,loc.a_recordtypename_c#loc.days_open,
  )
SELECT *
FROM labeled_opportunities_click_text
WHERE ARRAY_LENGTH(a_is_won) > 0
