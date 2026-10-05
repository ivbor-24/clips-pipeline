import { test, expect } from '@playwright/test';

// The backend decides the mode: without API_PASSWORD there is no login page;
// with it, set E2E_PASSWORD to the same value to run the password tests.
const password = process.env.E2E_PASSWORD;

test.describe('Access without a password', () => {
  test.skip(!!password, 'backend runs with API_PASSWORD');

  test('should open the dashboard without login', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveURL('/dashboard');
  });

  test('should skip the login page', async ({ page }) => {
    await page.goto('/login');
    await expect(page).toHaveURL('/dashboard');
  });
});

test.describe('Access with the shared password', () => {
  test.skip(!password, 'set E2E_PASSWORD to the backend API_PASSWORD');

  test('should redirect to login when not signed in', async ({ page }) => {
    await page.goto('/dashboard');
    await expect(page).toHaveURL('/login');
  });

  test('should reject a wrong password', async ({ page }) => {
    await page.goto('/login');
    await page.getByPlaceholder('Password').fill('wrong-password');
    await page.getByRole('button', { name: 'Sign in' }).click();
    await expect(page.getByText('Wrong password')).toBeVisible();
  });

  test('should sign in with the password', async ({ page }) => {
    await page.goto('/login');
    await page.getByPlaceholder('Password').fill(password!);
    await page.getByRole('button', { name: 'Sign in' }).click();
    await expect(page).toHaveURL('/dashboard');
  });
});
