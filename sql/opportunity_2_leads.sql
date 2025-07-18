-- join expanded-nebula-754.SFDC:lead and expanded-nebula-754.SFDC:opportunities_as  on ConvertedAccountId and opportunity_id
-- and select   is_closed and is_won from opportunities_as
SELECT
  t2.Email,
  t1.opportunity_id,
  t1.is_closed,
  t1.is_won
FROM
  `expanded-nebula-754`.`SFDC`.`opportunities_as` AS t1
INNER JOIN
  `expanded-nebula-754`.`SFDC`.`lead` AS t2
ON
  SAFE_CAST(TRIM(LOWER(t1.opportunity_id)) AS STRING) = SAFE_CAST(TRIM(LOWER(t2.ConvertedOpportunityId)) AS STRING)
 #WHERE t1.std_email_address IS NOT NULL AND t2.ConvertedOpportunityId IS NOT NULL AND t2.std_email_address IS NOT NULL;#expanded-nebula-754.sanbox_cdow.email_from_leads
 WHERE t1.opportunity_id IS NOT NULL  AND t2.ConvertedOpportunityId IS NOT NULL AND t2.Email IS NOT NULL;#expanded-nebula-754.sanbox_cdow.email_from_leads

  