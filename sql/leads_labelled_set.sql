


--CREATE OR REPLACE TABLE `expanded-nebula-754`.`sandbox_crdow`. `leads_dataset` AS
WITH golden_data AS(

  SELECT 
    std_email_address,
    CASE
      WHEN COUNTIF(search_term IS NOT NULL) > 0 THEN ARRAY_AGG(DISTINCT search_term IGNORE NULLS)
      ELSE ['']
    END AS search_terms,  
    CASE
      WHEN COUNTIF(source_campaign IS NOT NULL) > 0 THEN ARRAY_AGG(DISTINCT source_campaign IGNORE NULLS)
      ELSE ['']
    END AS source_campaigns,  
    CASE
      WHEN COUNTIF(department IS NOT NULL) > 0 THEN ARRAY_AGG(DISTINCT department IGNORE NULLS)
      ELSE ['']
    END AS departments,  
    CASE
      WHEN COUNTIF(sfdc_product_of_greatest_interest_pogi IS NOT NULL) > 0 THEN ARRAY_AGG(DISTINCT sfdc_product_of_greatest_interest_pogi IGNORE NULLS)
      ELSE ['']
    END AS sfdc_product_of_greatest_interest_pogis,
    CASE
      WHEN COUNTIF(lead_source_most_recent IS NOT NULL) > 0 THEN ARRAY_AGG(DISTINCT lead_source_most_recent IGNORE NULLS)
      ELSE ['']
    END AS lead_source_most_recents,      
    SUM(IFNULL(avg_revenue_by_shiptocluster_sap,0)) + SUM(IFNULL(avg_revenue_by_soldtocluster_sap,0)) AS av_revenue,
    SUM(IFNULL(total_revenue_webtrans,0)) + SUM(IFNULL(total_revenue_sap,0)) AS revenue,
    SUM(IFNULL(total_pageviews_count,0)) AS page_views,
  FROM `expanded-nebula-754.sanbox_cdow.golden_records`
    WHERE std_email_address IS NOT NULL 
    GROUP BY std_email_address

),

leads_email AS (
  SELECT
  
  Email AS Email,
  CASE
  WHEN COUNTIF(ConvertedOpportunityId IS NOT NULL) > 0 THEN ARRAY_AGG(DISTINCT ConvertedOpportunityId IGNORE NULLS)
  ELSE [''] 
  END AS ConvertedOpportunityIds

  FROM `expanded-nebula-754.SFDC.lead` 
  WHERE Email IS NOT NULL AND ConvertedOpportunityId != ''
  GROUP BY Email
),
opportunities AS (
  SELECT
  opportunity_id,
  stage AS op_stage,
  is_won,
  is_closed,
  stage,
  Type,
  campaign_name
  FROM
  `expanded-nebula-754`.`SFDC`.`opportunities_as` WHERE opportunity_id IS NOT NULL
),

lead_with_golden_record AS (
  SELECT
    t2.Email,
    t2.ConvertedOpportunityIds,
    t1.search_terms,
    t1.source_campaigns,
    t1.departments,
    t1.sfdc_product_of_greatest_interest_pogis,
    t1.lead_source_most_recents,
    t1.av_revenue,
    t1.page_views,
    t1.revenue
  FROM
    golden_data AS t1
  
  INNER JOIN
    leads_email AS t2
  ON
    t1.std_email_address = t2.Email  
),


labelled_opportunities AS (
  SELECT
    
    opp_id,
    t1.Email,
    t1.search_terms,
    t1.source_campaigns,
    t1.departments,
    t1.sfdc_product_of_greatest_interest_pogis,
    t1.lead_source_most_recents,
    t1.page_views,
    t1.revenue,
    t1.av_revenue,   
    --t2.op_stage,
    t2.is_won,
    t2.is_closed,
    t2.stage,
    t2.Type,
    t2.campaign_name,

  FROM
    lead_with_golden_record AS t1,
    UNNEST(t1.ConvertedOpportunityIds) AS opp_id
  JOIN
    opportunities AS t2
  --ON t2.opportunity_id = t1.ConvertedOpportunityIds
  ON t2.opportunity_id = opp_id
),
sfmc_clicks AS (
  SELECT 
  subscriberkey AS email,

    CASE
      WHEN COUNTIF(url IS NOT NULL) > 0 THEN ARRAY_AGG(DISTINCT url IGNORE NULLS)
      ELSE ['']
    END AS url,
    
    CASE
      WHEN COUNTIF(eventdate IS NOT NULL) > 0 THEN ARRAY_AGG(DISTINCT eventdate IGNORE NULLS)
      ELSE []
    END AS eventdate


  FROM
  `expanded-nebula-754.SFMC.sfmc_click`
  WHERE subscriberkey IS NOT NULL
  GROUP BY subscriberkey
),

sfmc_click AS (
  SELECT 
  subscriberkey AS email,
  url,
  eventdate,
  FROM
  `expanded-nebula-754.SFMC.sfmc_click`
  WHERE subscriberkey IS NOT NULL

),

labelled_opportunities_click AS (
  SELECT

#    ARRAY_TO_STRING(t1.search_terms,',') AS search_terms,
#    ARRAY_TO_STRING(t1.source_campaigns,',') AS source_campaigns,
#    ARRAY_TO_STRING(t1.departments,',') AS departments,  
    t1.opp_id,
    t1.Email,
    ARRAY_TO_STRING(t2.url,',') AS url,
    #ARRAY_TO_STRING(t2.eventdate,',') AS evendate,
    
    ARRAY_TO_STRING(
      ARRAY(
        SELECT CAST(ts AS STRING)
        FROM UNNEST(t2.eventdate) AS ts
      ),
      ','
    ) AS eventdate,

    t1.search_terms,
    t1.source_campaigns,
    t1.departments,
    t1.sfdc_product_of_greatest_interest_pogis,
    t1.lead_source_most_recents,
    t1.page_views,
    t1.revenue,
    t1.av_revenue,   
    --t2.op_stage,
    t1.is_won,
    t1.is_closed,
    t1.stage,
    t1.Type,
    t1.campaign_name,

  FROM labelled_opportunities AS t1
  LEFT JOIN sfmc_clicks AS t2
  
  
  ON t2.email = t1.Email

),


labelled_opportunities_test AS (
  SELECT
    opp_id,
    t1.Email
  FROM
    lead_with_golden_record AS t1,
    UNNEST(t1.ConvertedOpportunityIds) AS opp_id
  JOIN
    opportunities AS t2
  ON t2.opportunity_id = opp_id
),

email_check AS (
  SELECT
    opp_id,
    COUNT(*) AS total_emails,
    COUNT(DISTINCT Email) AS distinct_emails,
    COUNTIF(Email IS NULL OR Email = '') AS null_or_empty_emails
  FROM labelled_opportunities
  GROUP BY opp_id
  HAVING COUNT(*) > 1
),
duplicates_check AS (
  SELECT
    Email,
    ConvertedOpportunityIds,
    COUNT(*) AS total_elements,
    COUNT(DISTINCT id) AS distinct_elements
  FROM lead_with_golden_record,
  UNNEST(ConvertedOpportunityIds) AS id
  GROUP BY Email, ConvertedOpportunityIds
)

#SELECT
#  subscriberkey AS email,
#  COUNT(*) AS occurrence_count
#FROM `expanded-nebula-754.SFMC.sfmc_click`
#WHERE subscriberkey IS NOT NULL
#GROUP BY subscriberkey
#HAVING COUNT(*) > 1
#ORDER BY occurrence_count DESC

#SELECT * FROM sfmc_click

#SELECT
#  email AS email,
#  url,
#  eventdate,
  
#FROM sfmc_clicks
#WHERE email IS NOT NULL


SELECT * FROM labelled_opportunities_click




