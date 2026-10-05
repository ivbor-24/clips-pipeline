import { test, expect } from '@playwright/test';

test.describe('Config', () => {
  test('should show settings editor', async ({ page }) => {
    await page.goto('/config');
    await expect(page.getByRole('heading', { name: 'Settings' })).toBeVisible();
  });

  test('should show settings groups', async ({ page }) => {
    await page.goto('/config');
    await expect(page.getByRole('button', { name: 'Save Changes' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Reset all' })).toBeVisible();
  });
});
